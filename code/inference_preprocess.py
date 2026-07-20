import os
import logging
import re
import numpy as np
import tifffile
import zarr
from distributed import Client, LocalCluster
import dask.array as da
from dask_image.ndfilters import gaussian_filter as gaussian_filter_dask
from pathlib import Path
from masking import get_mask
from estimate_background import background_estimation
from utils import load_tile_paths, parse_tile_path, tile_output_prefix
import argparse

# -------- CONFIG ---------
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess zarr arrays for inference.")
    parser.add_argument('--zarr', type=str, help='Path to the parent zarr group.')
    parser.add_argument('--tile-json', type=str, help='JSON file containing a tile_paths list of tile zarr paths. If set, only those tile paths are processed.')
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


def discover_parent_zarr_arrays(args):
    print(f"Opening zarr group: {args.zarr}")
    zarr_group = zarr.open_group(args.zarr, mode='r')
    dataset_name = Path(args.zarr).parent.name

    # Discover tile groups for the requested channel (matches both v2
    # ``..._ch_488.zarr`` and v3 ``..._ch_488.ome.zarr`` naming).
    tile_re = re.compile(rf"_ch_{re.escape(args.channel)}(?:\.ome)?\.zarr$")
    array_specs = []
    for tile_group in zarr_group.keys():
        if tile_re.search(tile_group):
            array_specs.append(
                (
                    tile_group,
                    zarr_group[tile_group][str(args.res)],
                    f"{dataset_name}_{tile_group}",
                )
            )
    if not array_specs:
        print("No arrays found matching your channel/resolution pattern.")

    return array_specs


def discover_tile_json_arrays(args):
    tile_paths = load_tile_paths(args.tile_json)
    array_specs = []

    for tile_path in tile_paths:
        _tile_name, tile_channel = parse_tile_path(tile_path)
        if tile_channel != args.channel:
            continue

        print(f"Opening selected tile zarr: {tile_path}")
        try:
            tile_group = zarr.open_group(tile_path, mode='r')
            arr_proxy = tile_group[str(args.res)]
        except Exception as exc:
            raise ValueError(
                f"Could not open resolution {args.res} from selected tile zarr "
                f"{tile_path}: {exc}"
            ) from exc

        array_specs.append(
            (
                tile_path,
                arr_proxy,
                tile_output_prefix(tile_path),
            )
        )

    if not array_specs:
        raise ValueError(
            f"Tile JSON {args.tile_json} contains no tile_paths for channel {args.channel}."
        )

    return array_specs


def process_and_save_array(array_path: str, arr_proxy, output_prefix, raw_save_dir, mask_save_dir, gaussian_sigma, threshold_method, fixed_threshold):
    """Process a single array in the group."""
    print(f"\nProcessing {array_path} ...")
    # Load with dask (aligning blocks to the on-disk zarr/shard chunking) and
    # squeeze to remove degenerate dims.
    arr = da.from_zarr(arr_proxy).astype(np.float32).squeeze()
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
    raw_out = os.path.join(raw_save_dir, f"{output_prefix}_data.tif")
    mask_out = os.path.join(mask_save_dir, f"{output_prefix}_mask.tif")
    tifffile.imwrite(raw_out, arr_corrected, imagej=True, compression='zlib')
    tifffile.imwrite(mask_out, mask.astype(np.uint8), imagej=True, compression='zlib')
    print(f"  Saved raw: {raw_out}\n  Saved mask: {mask_out}")

def main():
    args = parse_args()
    os.makedirs(args.raw_save_dir, exist_ok=True)
    os.makedirs(args.mask_save_dir, exist_ok=True)
    processed = 0

    client = Client(LocalCluster(processes=False))

    try:
        if args.tile_json:
            array_specs = discover_tile_json_arrays(args)
        else:
            array_specs = discover_parent_zarr_arrays(args)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    if not array_specs:
        return

    print(f"Found {len(array_specs)} arrays to process.")

    for array_path, arr_proxy, output_prefix in array_specs:
        try:
            process_and_save_array(
                array_path, arr_proxy, output_prefix, args.raw_save_dir,
                args.mask_save_dir, args.gaussian_sigma, args.threshold_method,
                args.fixed_threshold
            )
            processed += 1
        except Exception:
            logging.exception(f"  ERROR processing {array_path}")

    print(f"\nDONE. Processed {processed}/{len(array_specs)} arrays.")

if __name__ == "__main__":
    main()
