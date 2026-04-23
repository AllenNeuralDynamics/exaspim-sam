import os
import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, label
from skimage.morphology import ball
import argparse
from tqdm import tqdm
import json
import re

STAT_CHANNELS = ("488", "561")


def detect_channel_from_filename(filename: str) -> str:
    match = re.search(r"_ch_(488|561)(?=(_|\.))", filename)
    if not match:
        raise ValueError(
            f"Could not detect channel token in filename '{filename}'. "
            "Expected '_ch_488' or '_ch_561'."
        )
    return match.group(1)


def build_raw_channel_filename(filename: str, source_channel: str, target_channel: str) -> str:
    raw_filename = filename.replace("_pred.tif", "_data.tif").replace("_pred.tiff", "_data.tiff")
    return raw_filename.replace(f"_ch_{source_channel}", f"_ch_{target_channel}", 1)


def get_raw_paths_from_pred(pred_path: str, raw_base_dir: str) -> dict:
    """
    Constructs paths to corresponding raw data files for channels 488 and 561
    from a single prediction file path, regardless of inference channel.
    """
    pred_filename = os.path.basename(pred_path)
    source_channel = detect_channel_from_filename(pred_filename)

    return {
        channel: os.path.join(
            raw_base_dir,
            channel,
            build_raw_channel_filename(pred_filename, source_channel, channel),
        )
        for channel in STAT_CHANNELS
    }


def border_protected_closing(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    selem = ball(radius)

    # 1) Dilate
    D = binary_dilation(mask, structure=selem)

    # 2) Make a “border band” of width ~radius
    border = np.zeros_like(mask, dtype=bool)
    border[[0, -1], :, :] = True
    border[:, [0, -1], :] = True
    border[:, :, [0, -1]] = True
    band = binary_dilation(border, structure=selem)

    # 3) Erode everywhere, then keep original (dilated) values inside the band
    E = binary_erosion(D, structure=selem)
    out = np.where(band, D, E)  # (E & ~band) | (D & band)

    return out


def size_filter(
    mask: np.ndarray,
    min_size: int | None = None,
    k_largest: int | None = None,
) -> np.ndarray:
    """
    Optionally filter connected components by size and/or keep the K largest.

    Rules:
    - If min_size is None and k_largest is None: return mask unchanged.
    - If min_size is set and no components satisfy it: raise ValueError.
    - If k_largest is set and min_size is also set: keep the top-K among
      components >= min_size; if none satisfy, raise ValueError.
    - If k_largest is set and min_size is None: keep the top-K among all
      components (ignoring background).
    - If only min_size is set: keep all components >= min_size; if none, raise.
    """
    # No size filtering requested
    if min_size is None and k_largest is None:
        return mask

    # Label connected components in the filled mask.
    labeled_array, num_features = label(mask)
    if num_features == 0:
        raise ValueError("No connected components found in the image.")

    if num_features < 2**16:
        labeled_array = labeled_array.astype(np.uint16)

    # Calculate sizes for each component (ignore background label 0).
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # ignore background

    if k_largest is not None:
        if k_largest <= 0:
            raise ValueError("k_largest must be a positive integer")

        # Choose candidates by min_size if provided
        if min_size is not None:
            candidate_labels = np.where(component_sizes >= min_size)[0]
            candidate_labels = candidate_labels[candidate_labels != 0]
            if candidate_labels.size == 0:
                raise ValueError(
                    "No connected components meet the specified min_size"
                )
        else:
            candidate_labels = np.where(component_sizes > 0)[0]

        candidate_sizes = component_sizes[candidate_labels]
        order = np.argsort(candidate_sizes)[::-1]
        keep_labels = candidate_labels[order][:k_largest]
        final_mask = np.isin(labeled_array, keep_labels)
    else:
        # Only min_size is set
        valid_labels = np.where(component_sizes >= min_size)[0]
        valid_labels = valid_labels[valid_labels != 0]
        if len(valid_labels) == 0:
            raise ValueError(
                "No connected components meet the specified min_size"
            )
        final_mask = np.isin(labeled_array, valid_labels)

    return final_mask


def post_process_mask(
    input_path: str,
    output_path: str,
    dilation_kernel_size: int,
    raw_base_dir: str = None,
    median_dict: dict = None
):
    """
    Loads a 3D mask, calculates median intensity on both raw channels,
    then performs post-processing (dilation, hole-filling) and saves the result.
    """
    try:
        # 1. Load the 3D prediction mask
        mask_vol = tifffile.imread(input_path).astype(bool)
        mask_vol = size_filter(mask_vol, min_size=10000, k_largest=2)

        # 2. Median intensity calculation using the original, non-dilated mask
        if raw_base_dir and median_dict is not None:
            raw_paths = get_raw_paths_from_pred(input_path, raw_base_dir)
            
            if np.any(mask_vol):
                for channel, raw_path in raw_paths.items():
                    if os.path.exists(raw_path):
                        raw_vol = tifffile.imread(raw_path)
                        non_zero_mask = mask_vol & (raw_vol > 0)
                        if np.any(non_zero_mask):
                            median_val = float(np.percentile(raw_vol[non_zero_mask], 75))
                            median_dict.setdefault(channel, []).append(median_val)
                    else:
                        print(f"  WARNING: Raw file for channel {channel} not found: {raw_path}")

        if dilation_kernel_size >= 1:
            print(f"dilating mask with radius {dilation_kernel_size}")
            mask_vol = border_protected_closing(mask_vol, dilation_kernel_size)

        # 4. Save the processed mask
        tifffile.imwrite(output_path, mask_vol.astype(np.uint8), compression='zlib', imagej=True)

    except Exception as e:
        print(f"  ERROR: Failed to process {os.path.basename(input_path)}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process 3D segmentation masks with dilation, hole filling, and intensity analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_path",
        default="/results/pred",
        help="Path to the input mask TIFF file or a directory containing prediction mask files."
    )
    parser.add_argument(
        "--output_dir",
        default="/results/postprocessed",
        help="Directory to save the processed mask files."
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Size of the cubic kernel for 3D dilation. e.g., 3 creates a 3x3x3 kernel. Use 1 to skip dilation."
    )
    parser.add_argument(
        "--raw_base_dir",
        default="/results/raw",
        help="Base directory for raw TIFF files, organized by channel subdirectories (e.g., /results/raw/488)."
    )
    parser.add_argument(
        "--save_json",
        default="median_intensity_summary.json",
        help="Filename for saving the median intensity summary."
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    median_dict = {}

    if os.path.isdir(args.input_path):
        print(f"Processing all TIFF files in directory: {args.input_path}")
        input_files = [
            f for f in os.listdir(args.input_path)
            if f.lower().endswith(('.tif', '.tiff'))
        ]
        
        if not input_files:
            print("No TIFF files found in the input directory. Exiting.")
            exit()
            
        print(f"Found {len(input_files)} files to process.")
        
        for filename in tqdm(input_files, desc="Processing masks"):
            input_file_path = os.path.join(args.input_path, filename)
            output_file_path = os.path.join(args.output_dir, filename)
            
            post_process_mask(
                input_file_path, 
                output_file_path, 
                args.kernel_size,
                raw_base_dir=args.raw_base_dir,
                median_dict=median_dict
            )

    elif os.path.isfile(args.input_path):
        print(f"Processing single file: {args.input_path}")
        output_file_path = os.path.join(args.output_dir, os.path.basename(args.input_path))

        post_process_mask(
            args.input_path, 
            output_file_path, 
            args.kernel_size,
            raw_base_dir=args.raw_base_dir,
            median_dict=median_dict
        )

    else:
        print(f"Error: Input path '{args.input_path}' is not a valid file or directory.")
        exit(1)

    # --- Compute mean of medians and save to JSON ---
    summary = {}
    for ch, medians in median_dict.items():
        if medians:
            summary[ch] = {
                "tile_medians": medians,
                "mean_of_medians": float(np.percentile(medians, 50))
            }
    
    if summary:
        json_path = os.path.join(args.output_dir, args.save_json)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\nMedian intensity summary saved to: {json_path}")
    else:
        print("\nNo median intensity data was calculated.")

    print("\nProcessing complete.")
    print(f"Processed masks are saved in: {args.output_dir}")
