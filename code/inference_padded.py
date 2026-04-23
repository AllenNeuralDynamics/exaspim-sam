import os
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from scipy.ndimage import find_objects
import gc

def get_bbox_from_mask(mask_2d):
    """Gets the bounding box from a 2D boolean mask."""
    if not np.any(mask_2d): return None
    s = find_objects(mask_2d)
    if not s: return None
    s = s[0]
    # find_objects returns slice objects; .start is inclusive, .stop is exclusive
    ymin, ymax = s[0].start, s[0].stop
    xmin, xmax = s[1].start, s[1].stop
    # Return as [xmin, ymin, xmax_inclusive, ymax_inclusive]
    return np.array([xmin, ymin, xmax - 1, ymax - 1], dtype=np.int32)

def pad_and_expand_bbox(raw_slice, bbox, expansion_pixels):
    """
    Pads an image slice to accommodate an expanded bounding box.
    
    Args:
        raw_slice (np.ndarray): The 2D image slice.
        bbox (np.ndarray): The original bounding box [xmin, ymin, xmax, ymax].
        expansion_pixels (int): The number of pixels to expand the box by on all sides.
        
    Returns:
        tuple: A tuple containing:
            - padded_slice (np.ndarray): The padded image.
            - padded_bbox (np.ndarray): The bbox coordinates relative to the padded image.
            - crop_info (tuple): (pad_top, pad_left, H, W) for cropping back later.
    """
    H, W = raw_slice.shape
    
    # 1. Expand the bounding box
    expanded_bbox = np.array([
        bbox[0] - expansion_pixels,
        bbox[1] - expansion_pixels,
        bbox[2] + expansion_pixels,
        bbox[3] + expansion_pixels
    ], dtype=np.int32)
    
    # 2. Calculate necessary padding for the image
    pad_left = max(0, -expanded_bbox[0])
    pad_top = max(0, -expanded_bbox[1])
    pad_right = max(0, expanded_bbox[2] - (W - 1))
    pad_bottom = max(0, expanded_bbox[3] - (H - 1))
    
    # 3. Pad the image slice
    # The raw_slice is normalized to [0,1], so padding with 0 is appropriate.
    padded_slice = np.pad(
        raw_slice,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='median',
        #constant_values=0
    )
    
    # 4. Adjust bbox coordinates to be relative to the new padded image
    padded_bbox = np.array([
        expanded_bbox[0] + pad_left,
        expanded_bbox[1] + pad_top,
        expanded_bbox[2] + pad_left,
        expanded_bbox[3] + pad_top
    ], dtype=np.float32)
    
    crop_info = (pad_top, pad_left, H, W)
    
    return padded_slice, padded_bbox, crop_info


def normalize_volume_by_percentiles(vol, p_low=0.5, p_high=99.5):
    low_val = np.percentile(vol, p_low)
    high_val = np.percentile(vol, p_high)
    print(f"Volume normalization percentiles: {p_low}th={low_val:.3f}, {p_high}th={high_val:.3f}")
    vol_norm = (np.clip(vol, low_val, high_val, dtype=np.float32) - low_val) / (high_val - low_val + 1e-8)
    return vol_norm.astype(np.float32)

def preprocess_slice(slice_np, target_size=1024):
    # Input slice_np assumed normalized to [0,1]
    if slice_np.ndim == 2:
        slice_norm_3c = np.stack([slice_np]*3, axis=0)
    elif slice_np.ndim == 3 and slice_np.shape[2] == 1:
        slice_norm_3c = np.repeat(slice_np.transpose(2,0,1), 3, axis=0)
    elif slice_np.ndim == 3 and slice_np.shape[2] == 3:
        slice_norm_3c = slice_np.transpose(2,0,1)
    else:
        raise ValueError(f"Unexpected slice shape for preprocessing: {slice_np.shape}")

    tensor = torch.from_numpy(slice_norm_3c).unsqueeze(0)  # (1,3,H,W)
    tensor_resized = F.interpolate(tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
    tensor_resized = torch.clamp(tensor_resized, 0.0, 1.0)  # Clamp to ensure in [0,1]
    return tensor_resized.squeeze(0)  # (3, target_size, target_size)

@torch.no_grad()
def run_medsam_inference_batched(
    raw_tiff_path, mask_tiff_path, save_dir, medsam_model, device, batch_size=8,
    use_amp=False, bbox_prompt_type="mask", bbox_expansion_pixels=20
):
    os.makedirs(save_dir, exist_ok=True)
    raw_vol = tifffile.imread(raw_tiff_path)
    mask_vol = tifffile.imread(mask_tiff_path).astype(bool)

    if raw_vol.shape != mask_vol.shape:
        raise ValueError(f"Raw shape {raw_vol.shape} and mask shape {mask_vol.shape} don't match")

    print("Normalizing volume by 0.5 and 99.5 percentiles...")
    raw_vol = normalize_volume_by_percentiles(raw_vol, 0.5, 99.5)

    depth, H, W = raw_vol.shape
    medsam_input_size = 1024
    preds = np.zeros((depth, H, W), dtype=np.uint8)

    for start_idx in range(0, depth, batch_size):
        end_idx = min(start_idx + batch_size, depth)
        
        # Batch lists to hold data for each slice
        batch_padded_slices = []
        batch_padded_bboxes = []
        batch_crop_infos = []
        batch_valid_indices = []

        for idx in range(start_idx, end_idx):
            bbox = get_bbox_from_mask(mask_vol[idx])
            
            # Skip slices with no valid mask to derive a bbox from
            if bbox is None:
                preds[idx] = 0
                continue
            
            # **MODIFIED STEP**: Pad slice and expand bbox
            raw_slice = raw_vol[idx]
            padded_slice, padded_bbox, crop_info = pad_and_expand_bbox(
                raw_slice, bbox, bbox_expansion_pixels
            )

            batch_valid_indices.append(idx)
            batch_padded_slices.append(padded_slice)
            batch_padded_bboxes.append(padded_bbox)
            batch_crop_infos.append(crop_info)

        if not batch_padded_slices:
            continue

        # Preprocess all padded slices in the batch
        batch_tensor = torch.stack([preprocess_slice(s) for s in batch_padded_slices]).to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            # 1. Get image embeddings
            emb = medsam_model.image_encoder(batch_tensor)

            # 2. Prepare bounding box prompts
            boxes_1024 = []
            for i, padded_bbox in enumerate(batch_padded_bboxes):
                # **MODIFIED STEP**: Scale bbox using the dimensions of the PADDED slice
                padded_H, padded_W = batch_padded_slices[i].shape
                xmin, ymin, xmax, ymax = padded_bbox
                scaled_box = np.array([
                    xmin / padded_W * medsam_input_size,
                    ymin / padded_H * medsam_input_size,
                    xmax / padded_W * medsam_input_size,
                    ymax / padded_H * medsam_input_size
                ], dtype=np.float32)
                boxes_1024.append(scaled_box)
            boxes_1024 = torch.tensor(boxes_1024, device=device).unsqueeze(1)

            # 3. Get prompt embeddings
            sparse_emb, dense_emb = medsam_model.prompt_encoder(points=None, boxes=boxes_1024, masks=None)
            
            # 4. Decode masks
            low_res_logits, _ = medsam_model.mask_decoder(
                image_embeddings=emb,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False
            )

        # 5. Post-process predictions
        for i in range(low_res_logits.shape[0]):
            logit = low_res_logits[i:i+1]
            padded_slice = batch_padded_slices[i]
            crop_info = batch_crop_infos[i]
            
            # **MODIFIED STEP**: Upsample to the PADDED dimensions
            padded_H, padded_W = padded_slice.shape
            pred_prob_padded = torch.sigmoid(F.interpolate(logit, size=(padded_H, padded_W), mode='bicubic', align_corners=False))
            pred_mask_padded = (pred_prob_padded.squeeze().cpu().numpy() > 0.5)
            
            # **MODIFIED STEP**: Crop the prediction back to the original image size
            pad_top, pad_left, orig_H, orig_W = crop_info
            pred_mask = pred_mask_padded[pad_top : pad_top + orig_H, pad_left : pad_left + orig_W]

            preds[batch_valid_indices[i]] = pred_mask.astype(np.uint8)

        del batch_tensor, emb, sparse_emb, dense_emb, low_res_logits, pred_prob_padded
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Processed slices {start_idx} to {end_idx-1}")

    save_path = os.path.join(save_dir, os.path.basename(raw_tiff_path).replace(".tif", "_pred.tif"))
    tifffile.imwrite(save_path, preds, compression='zlib', imagej=True, metadata={'axes': 'ZYX'})
    print(f"Saved predictions to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_tiff_path", default="/data/mask-result-2025-06-02/raw")
    parser.add_argument("--mask_tiff_path", default="/data/mask-result-2025-06-02/mask")
    parser.add_argument("--pred_save_dir", default="/results/pred")
    parser.add_argument("--medsam_checkpoint", default="/data/mask-result-2025-06-02/medsam_best.pth")
    parser.add_argument("--medsam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    # **NEW ARGUMENT**
    parser.add_argument(
        "--bbox_expansion_pixels", 
        type=int, 
        default=32, 
        help="Number of pixels to expand bounding box prompts and pad the image."
    )
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--use_torch_compile", action="store_true", default=True, help="Use torch.compile to optimize model")
    # I removed the `bbox_prompt_type` argument as this code now always uses the mask-derived bbox.
    # You can add it back if you need to switch between behaviors.
    args = parser.parse_args()

    device = torch.device(args.device)
    medsam = sam_model_registry[args.medsam_model_type](checkpoint=args.medsam_checkpoint).to(device)

    if args.use_torch_compile and hasattr(torch, 'compile'):
        print("Applying torch.compile() to MedSAM model...")
        try:
            medsam = torch.compile(medsam, mode="reduce-overhead")
            print("torch.compile() successful.")
        except Exception as e:
            print(f"torch.compile() failed: {e}. Continuing without compile.")

    medsam.eval()

    def is_tiff(f):
        return f.lower().endswith('.tif') or f.lower().endswith('.tiff')

    if os.path.isdir(args.raw_tiff_path) or os.path.isdir(args.mask_tiff_path):
        raw_dir = args.raw_tiff_path if os.path.isdir(args.raw_tiff_path) else os.path.dirname(args.raw_tiff_path)
        mask_dir = args.mask_tiff_path if os.path.isdir(args.mask_tiff_path) else os.path.dirname(args.mask_tiff_path)
        raw_files = [f for f in os.listdir(raw_dir) if is_tiff(f) and f.endswith('_data.tif')]
        mask_files = set([f for f in os.listdir(mask_dir) if is_tiff(f) and f.endswith('_mask.tif')])

        print(f"Scanning for raw/mask pairs in:\n  raw: {raw_dir}\n  mask: {mask_dir}")

        n_processed = 0
        for raw_fn in raw_files:
            base_prefix = raw_fn.replace('_data.tif', '')
            mask_fn = f"{base_prefix}_mask.tif"
            if mask_fn not in mask_files:
                print(f"  No mask for {raw_fn}, skipping.")
                continue

            raw_path = os.path.join(raw_dir, raw_fn)
            mask_path = os.path.join(mask_dir, mask_fn)
            print(f"Processing:\n  RAW:  {raw_path}\n  MASK: {mask_path}")
            try:
                run_medsam_inference_batched(
                    raw_path,
                    mask_path,
                    args.pred_save_dir,
                    medsam,
                    device,
                    batch_size=args.batch_size,
                    use_amp=args.use_amp,
                    bbox_expansion_pixels=args.bbox_expansion_pixels, # Pass the new arg
                )
                n_processed += 1
            except Exception as e:
                print(f"  ERROR: Failed processing pair {raw_fn} / {mask_fn}: {e}")

        print(f"Batch processing complete: {n_processed} pairs processed.")
    else:
        run_medsam_inference_batched(
            args.raw_tiff_path,
            args.mask_tiff_path,
            args.pred_save_dir,
            medsam,
            device,
            batch_size=args.batch_size,
            use_amp=args.use_amp,
            bbox_expansion_pixels=args.bbox_expansion_pixels, # Pass the new arg
        )