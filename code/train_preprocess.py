import os
import numpy as np
import tifffile
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import SimpleITK as sitk

def clip_and_normalize_intensity(volume, lower_percentile=0.5, upper_percentile=99.5):
    vol_float = volume.astype(np.float32)
    low = np.percentile(vol_float, lower_percentile)
    high = np.percentile(vol_float, upper_percentile)
    if low >= high:
        low, high = vol_float.min(), vol_float.max()
    clipped = np.clip(vol_float, low, high)
    if high == low:
        return np.zeros_like(clipped)
    else:
        return (clipped - low) / (high - low)

def resize_image(image, size, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    resized = cv2.resize(image, (size, size), interpolation=interp)
    return resized

def process_and_save_slice(args):
    img_slice_2d, gt_slice_2d, target_size, base_path, slice_name, skip_empty = args
    if skip_empty and not np.any(gt_slice_2d):
        return None

    # Repeat grayscale slice to 3-channel float32
    img_3c = np.repeat(img_slice_2d[:, :, None], 3, axis=-1).astype(np.float32)
    resized_img = resize_image(img_3c, target_size, is_mask=False)

    min_val, max_val = resized_img.min(), resized_img.max()
    if max_val > min_val:
        resized_img = (resized_img - min_val) / (max_val - min_val)
    else:
        resized_img = np.zeros_like(resized_img)

    resized_gt = resize_image(gt_slice_2d, target_size, is_mask=True).astype(np.uint8)

    if resized_img.shape[:2] != resized_gt.shape:
        print(f"Shape mismatch for {slice_name}: img {resized_img.shape}, gt {resized_gt.shape}")
        return None

    img_path = os.path.join(base_path, "imgs", slice_name + ".npy")
    gt_path = os.path.join(base_path, "gts", slice_name + ".npy")

    try:
        np.save(img_path, resized_img)
        np.save(gt_path, resized_gt)
        return f"Saved: {slice_name}"
    except Exception as e:
        return f"Error saving {slice_name}: {e}"

def main(args):
    npy_base_path = args.output_npy_path
    os.makedirs(os.path.join(npy_base_path, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(npy_base_path, "gts"), exist_ok=True)
    if args.save_sanity_checks:
        sanity_check_path = os.path.join(npy_base_path, "sanity_checks")
        os.makedirs(sanity_check_path, exist_ok=True)
        print(f"Sanity check NIfTI files will be saved to: {sanity_check_path}")

    raw_files = sorted([f for f in os.listdir(args.input_raw_dir) if f.endswith("_data.tif")])
    mask_files = sorted([f for f in os.listdir(args.input_mask_dir) if f.endswith("_mask.tif")])

    raw_map = {f.replace("_data.tif", ""): f for f in raw_files}
    file_pairs = []
    for mf in mask_files:
        base_name = mf.replace("_mask.tif", "")
        if base_name in raw_map:
            file_pairs.append((raw_map[base_name], mf))
        else:
            print(f"Warning: Mask file {mf} has no matching data file. Skipping.")

    print(f"Found {len(file_pairs)} data-mask pairs.")
    num_workers = args.num_workers if args.num_workers > 0 else os.cpu_count()
    print(f"Using {num_workers} worker processes.")

    stride = max(1, args.stride)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for raw_fname, mask_fname in file_pairs:
            base_name = raw_fname.replace("_data.tif", "")
            print(f"\nProcessing volume: {base_name}")

            try:
                raw_vol = tifffile.imread(os.path.join(args.input_raw_dir, raw_fname))
                gt_vol = tifffile.imread(os.path.join(args.input_mask_dir, mask_fname)).astype(np.uint8)
            except Exception as e:
                print(f"Failed to load {base_name}: {e}")
                continue

            if raw_vol.shape != gt_vol.shape:
                print(f"Shape mismatch for {base_name}, skipping.")
                continue

            gt_binary = (gt_vol > 0).astype(np.uint8)
            z_idxs = np.where(gt_binary.any(axis=(1, 2)))[0]
            if z_idxs.size == 0:
                print(f"No mask found in {base_name}, skipping.")
                continue

            raw_vol = clip_and_normalize_intensity(raw_vol)

            min_z, max_z = z_idxs.min(), z_idxs.max()
            raw_roi_norm = raw_vol[min_z:max_z+1]
            gt_roi = gt_binary[min_z:max_z+1]

            print(f"Cropped Z slices: {min_z} to {max_z}, depth: {raw_roi_norm.shape[0]}")

            if args.save_npz:
                npz_path = os.path.join(npy_base_path, args.output_prefix + base_name + '_3D_roi.npz')
                np.savez_compressed(npz_path, imgs=(raw_roi_norm * 255).astype(np.uint8), gts=gt_roi)

            if args.save_sanity_checks:
                img_sitk = sitk.GetImageFromArray((raw_roi_norm * 255).astype(np.uint8))
                gt_sitk = sitk.GetImageFromArray(gt_roi)
                sitk.WriteImage(img_sitk, os.path.join(sanity_check_path, args.output_prefix + base_name + "_img.nii.gz"))
                sitk.WriteImage(gt_sitk, os.path.join(sanity_check_path, args.output_prefix + base_name + "_gt.nii.gz"))

            d, h, w = raw_roi_norm.shape

            # XY slices (standard)
            for i in range(0, d, stride):
                img_slice = raw_roi_norm[i]
                gt_slice = gt_roi[i]
                slice_name = f"{args.output_prefix}{base_name}-xy-{str(i).zfill(3)}"
                task_args = (img_slice, gt_slice, args.image_size, npy_base_path, slice_name, args.skip_empty_gt_slices)
                futures.append(executor.submit(process_and_save_slice, task_args))

            # YZ slices (if enabled)
            if args.include_yz:
                for y in range(0, h, stride):
                    img_slice = raw_roi_norm[:, y, :]   # shape (depth, width)
                    gt_slice = gt_roi[:, y, :]
                    #img_slice = img_slice.T  # (width, depth)
                    #gt_slice = gt_slice.T
                    slice_name = f"{args.output_prefix}{base_name}-yz-{str(y).zfill(3)}"
                    task_args = (img_slice, gt_slice, args.image_size, npy_base_path, slice_name, args.skip_empty_gt_slices)
                    futures.append(executor.submit(process_and_save_slice, task_args))

            # XZ slices (if enabled)
            if args.include_xz:
                for x in range(0, w, stride):
                    img_slice = raw_roi_norm[:, :, x]   # shape (depth, height)
                    gt_slice = gt_roi[:, :, x]
                    #img_slice = img_slice.T  # (height, depth)
                    #gt_slice = gt_slice.T
                    slice_name = f"{args.output_prefix}{base_name}-xz-{str(x).zfill(3)}"
                    task_args = (img_slice, gt_slice, args.image_size, npy_base_path, slice_name, args.skip_empty_gt_slices)
                    futures.append(executor.submit(process_and_save_slice, task_args))

        if futures:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Saving slices"):
                result = future.result()
                if result and "Error" in result:
                    print(result)

    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_raw_dir', required=True)
    parser.add_argument('--input_mask_dir', required=True)
    parser.add_argument('--output_npy_path', required=True)
    parser.add_argument('--output_prefix', default="spim_")
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--save_npz', action='store_true')
    parser.add_argument('--save_sanity_checks', action='store_true')
    parser.add_argument('--skip_empty_gt_slices', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--stride', type=int, default=1, help='Stride for slice extraction on each axis')
    parser.add_argument('--include_yz', action='store_true', help='Also extract YZ slices')
    parser.add_argument('--include_xz', action='store_true', help='Also extract XZ slices')
    args = parser.parse_args()
    main(args)
