import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def collect_pngs_by_view(png_dir, view):
    files = [f for f in os.listdir(png_dir)
             if f.lower().endswith('.png') and f'_{view.lower()}_sanity_' in f.lower()]
    files_sorted = sorted(files)  # Assumes filenames are sorted in intended column-major order
    return [os.path.join(png_dir, f) for f in files_sorted]

def render_tile_grid(png_files, out_path=None, n_rows=4, n_cols=5, figsize=(15,12)):
    n_images = n_rows * n_cols
    files_to_show = png_files[:n_images]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Fill in column-major order
    for i, fname in enumerate(files_to_show):
        col = i // n_rows
        row = i % n_rows
        ax = axes[row, col]
        img = mpimg.imread(fname)
        ax.imshow(img, interpolation='nearest', aspect='auto')
        ax.axis('off')
    # Hide any unused axes
    for i in range(len(files_to_show), n_rows * n_cols):
        col = i // n_rows
        row = i % n_rows
        axes[row, col].axis('off')
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=150)
        print(f"Grid image saved to {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--png_dir', required=True, help="Directory containing PNG images")
    parser.add_argument('--output_dir', default=None, help="Directory to save the output grid PNGs (default: png_dir)")
    parser.add_argument('--n_rows', type=int, default=4, help="Number of rows in the grid")
    parser.add_argument('--n_cols', type=int, default=5, help="Number of columns in the grid")
    args = parser.parse_args()
    output_dir = args.output_dir or args.png_dir
    os.makedirs(output_dir, exist_ok=True)

    views = ["XY", "ZY", "ZX"]
    for view in views:
        pngs = collect_pngs_by_view(args.png_dir, view)
        if len(pngs) == 0:
            print(f"No PNGs found for view {view}")
            continue
        out_path = os.path.join(output_dir, f'grid_{view}.png')
        render_tile_grid(pngs, out_path=out_path, n_rows=args.n_rows, n_cols=args.n_cols)
