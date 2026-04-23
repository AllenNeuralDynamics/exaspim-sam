import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt

def normalize_img(img):
    """Normalize image for display."""
    # r_min, r_max = np.percentile(img, [1,99])
    r_min, r_max = 0, 100
    return np.clip((img - r_min) / (r_max - r_min + 1e-5), 0, 1)

def save_view(raw_view, mask_view, out_fn, mask_label, cmap):
    raw_disp = normalize_img(raw_view)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(raw_disp, cmap='gray', interpolation='nearest', aspect='equal')
    # Make sure mask is 0/1, and set vmin/vmax for imshow to use the full colormap range
    mask_norm = (mask_view > 0).astype(float)  # binary mask
    ax.imshow(np.ma.masked_where(mask_norm==0, mask_norm), cmap=cmap, vmin=0, vmax=1, alpha=0.5, interpolation='nearest', aspect='equal')
    ax.axis('off')
    plt.savefig(out_fn, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved sanity check image: {out_fn}")

def save_sanity_check_png(raw_path, mask_path, output_dir, num_middle_slices=1, mask_label="gt"):
    raw = tifffile.imread(raw_path)
    mask = tifffile.imread(mask_path)
    if raw.shape != mask.shape:
        print(f"Shape mismatch: {raw_path} {raw.shape} vs {mask_path} {mask.shape}")
        return
    basename = os.path.basename(raw_path).replace('_data.tif', '')
    z, y, x = raw.shape

    # Indices for XY view (slices along z)
    mid_z_indices = np.linspace(z//2 - num_middle_slices//2, z//2 + num_middle_slices//2, num=num_middle_slices, dtype=int)
    # Indices for ZY (slices along x)
    mid_x_indices = np.linspace(x//2 - num_middle_slices//2, x//2 + num_middle_slices//2, num=num_middle_slices, dtype=int)
    # Indices for ZX (slices along y)
    mid_y_indices = np.linspace(y//2 - num_middle_slices//2, y//2 + num_middle_slices//2, num=num_middle_slices, dtype=int)

    cmap = 'Reds' if mask_label == "gt" else 'Greens'
    label = "GT" if mask_label == "gt" else "PRED"

    # XY (Z-slice)
    for idx in mid_z_indices:
        if not (0 <= idx < z): continue
        out_fn = os.path.join(output_dir, f"{basename}_slice{idx:03d}_XY_sanity_{mask_label}.png")
        save_view(raw[idx], mask[idx], out_fn, mask_label, cmap)

    # ZY (X-slice)
    for idx in mid_x_indices:
        if not (0 <= idx < x): continue
        raw_view = raw[:, :, idx]    # (z, y)
        mask_view = mask[:, :, idx]
        out_fn = os.path.join(output_dir, f"{basename}_slice{idx:03d}_ZY_sanity_{mask_label}.png")
        save_view(raw_view, mask_view, out_fn, mask_label, cmap)

    # ZX (Y-slice)
    for idx in mid_y_indices:
        if not (0 <= idx < y): continue
        raw_view = raw[:, idx, :]    # (z, x)
        mask_view = mask[:, idx, :]
        out_fn = os.path.join(output_dir, f"{basename}_slice{idx:03d}_ZX_sanity_{mask_label}.png")
        save_view(raw_view, mask_view, out_fn, mask_label, cmap)

def main(raw_dir, mask_dir, output_dir, num_middle_slices=1):
    os.makedirs(output_dir, exist_ok=True)
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('_data.tif')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.tif')]
    pred_files = [f for f in os.listdir(mask_dir) if f.endswith('_pred.tif')]

    n = 0
    for raw_fn in raw_files:
        basename = raw_fn.replace('_data.tif', '')
        mask_fn = f"{basename}_mask.tif"
        pred_fn = f"{basename}_data_pred.tif"
        raw_path = os.path.join(raw_dir, raw_fn)
        if mask_fn in mask_files:
            mask_path = os.path.join(mask_dir, mask_fn)
            save_sanity_check_png(raw_path, mask_path, output_dir, num_middle_slices=num_middle_slices, mask_label="gt")
        if pred_fn in pred_files:
            pred_path = os.path.join(mask_dir, pred_fn)
            save_sanity_check_png(raw_path, pred_path, output_dir, num_middle_slices=num_middle_slices, mask_label="pred")
        if mask_fn in mask_files or pred_fn in pred_files:
            n += 1
        else:
            print(f"WARNING: No mask or prediction for {raw_fn}, skipping.")
    print(f"Sanity check done. {n} volumes processed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True, help='Directory containing *_data.tif files')
    parser.add_argument('--mask_dir', required=True, help='Directory containing *_mask.tif and/or *_pred.tif files')
    parser.add_argument('--output_dir', required=True, help='Output directory for sanity PNGs')
    parser.add_argument('--num_slices', type=int, default=1, help='Number of middle slices to visualize (default: 1)')
    args = parser.parse_args()
    main(args.raw_dir, args.mask_dir, args.output_dir, num_middle_slices=args.num_slices)
