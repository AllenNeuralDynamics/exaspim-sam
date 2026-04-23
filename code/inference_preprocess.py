import os
import logging
import numpy as np
import tifffile
import zarr
from distributed import Client, LocalCluster
import dask.array as da
from dask_image.ndfilters import gaussian_filter as gaussian_filter_dask
from pathlib import Path
from masking import get_mask
from estimate_background import background_estimation
import argparse

# -------- CONFIG ---------
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess zarr arrays for inference.")
    parser.add_argument('--zarr', type=str, default='s3://aind-open-data/exaSPIM_754615_2025-01-23_16-44-53/SPIM.ome.zarr', help='Path to the zarr group.')
    parser.add_argument('--channel', type=str, default='488', help='Channel to process.')
    parser.add_argument('--res', type=int, default=4, help='Resolution level (integer, 0=highest).')
    parser.add_argument('--raw-save-dir', type=str, default='/results/raw', help='Directory to save raw output.')
    parser.add_argument('--mask-save-dir', type=str, default='/results/mask', help='Directory to save mask output.')
    parser.add_argument('--gaussian-sigma', type=float, default=2, help='Sigma for Gaussian blur.')
    parser.add_argument('--threshold-method', type=str, default='fixed', choices=['median', 'fixed'], help='Thresholding method.')
    parser.add_argument('--fixed-threshold', type=float, default=1, help='Fixed threshold value (if method is fixed).')
    return parser.parse_args()

def infer_threshold(data: np.ndarray, method: str = "median", fixed_value=None) -> float:
    """Determine threshold value."""
    # data = data[data > 0]
    if method == "median":
        return float(np.median(data))
    elif method == "fixed" and fixed_value is not None:
        return float(fixed_value)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

def process_and_save_array(array_path: str, zarr_group, dataset_name, ch, res, raw_save_dir, mask_save_dir, gaussian_sigma, threshold_method, fixed_threshold):
    """Process a single array in the group."""
    print(f"\nProcessing {array_path} ...")
    arr_proxy = zarr_group[array_path][str(res)]
    # Load with dask and squeeze to remove degenerate dims
    arr = da.from_array(arr_proxy).astype(np.float32).squeeze()
    arr = gaussian_filter_dask(arr, sigma=gaussian_sigma).compute()
    print(f"  Shape after blur: {arr.shape}, dtype: {arr.dtype}")

    # Background estimation & subtraction
    bkg = background_estimation(arr)
    arr_corrected = arr - bkg
    orig_dtype = np.uint16
    arr_corrected = np.clip(arr_corrected, 0, 65535.0).astype(orig_dtype)
    print(f"  Background corrected. Value range: {arr_corrected.min()} - {arr_corrected.max()}")

    # Threshold
    threshold = infer_threshold(arr_corrected, threshold_method, fixed_threshold)
    print(f"  Threshold for mask: {threshold}")

    # Masking
    mask = arr_corrected > threshold
    if mask.shape != arr_corrected.shape:
        print(f"  WARNING: Mask shape {mask.shape} does not match array {arr_corrected.shape}. Skipping.")
        return

    # Save paths
    raw_out = os.path.join(raw_save_dir, f"{dataset_name}_{array_path}_data.tif")
    mask_out = os.path.join(mask_save_dir, f"{dataset_name}_{array_path}_mask.tif")
    tifffile.imwrite(raw_out, arr_corrected, imagej=True, compression='zlib')
    tifffile.imwrite(mask_out, mask.astype(np.uint8), imagej=True, compression='zlib')
    print(f"  Saved raw: {raw_out}\n  Saved mask: {mask_out}")

def main():
    args = parse_args()
    os.makedirs(args.raw_save_dir, exist_ok=True)
    os.makedirs(args.mask_save_dir, exist_ok=True)
    print(f"Opening zarr group: {args.zarr}")
    zarr_group = zarr.open_group(args.zarr, mode='r')
    dataset_name = Path(args.zarr).parent.name
    processed = 0

    client = Client(LocalCluster(processes=False))

    # Discover arrays with matching channel and resolution
    array_paths = []
    for tile_group in zarr_group.keys():
        if args.channel in tile_group:
            array_paths.append(f"{tile_group}")
    if not array_paths:
        print("No arrays found matching your channel/resolution pattern.")
        return

    print(f"Found {len(array_paths)} arrays to process.")

    for array_path in array_paths:
        try:
            process_and_save_array(
                array_path, zarr_group, dataset_name, args.channel, args.res,
                args.raw_save_dir, args.mask_save_dir, args.gaussian_sigma,
                args.threshold_method, args.fixed_threshold
            )
            processed += 1
        except Exception:
            logging.exception(f"  ERROR processing {array_path}")

    print(f"\nDONE. Processed {processed}/{len(array_paths)} arrays.")

if __name__ == "__main__":
    main()
