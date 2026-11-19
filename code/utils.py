import json
import re

import numpy as np


# Tile basenames come in two exaSPIM schemes: a flat index (``tile_000000``) and
# a grid-coordinate form (``tile_x_0000_y_0000_z_0000``). Keep this token as the
# single source of truth; it is reused for the anchored basename match below and
# for searching the token inside longer output filenames.
TILE_NAME = r"tile_(?:x_\d+_y_\d+_z_\d+|\d+)"
TILE_PATH_RE = re.compile(rf"^({TILE_NAME})_ch_(488|561)((?:\.ome)?\.zarr)$")
TILE_NAME_SEARCH_RE = re.compile(rf"({TILE_NAME})")


def load_tile_paths(tile_json_path: str) -> list[str]:
    try:
        with open(tile_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise ValueError(f"Could not read tile JSON {tile_json_path}: {exc}") from exc

    tile_paths = payload.get("tile_paths") if isinstance(payload, dict) else None
    if not isinstance(tile_paths, list):
        raise ValueError(
            f"Tile JSON {tile_json_path} must contain a list field named tile_paths."
        )

    if not tile_paths:
        raise ValueError(f"Tile JSON {tile_json_path} has no tile_paths entries.")

    invalid = [path for path in tile_paths if not isinstance(path, str) or not path.strip()]
    if invalid:
        raise ValueError(
            f"Tile JSON {tile_json_path} contains non-string or empty tile_paths entries."
        )

    return tile_paths


def parse_tile_path(tile_path: str) -> tuple[str, str]:
    basename = tile_path.rstrip("/").split("/")[-1]
    match = TILE_PATH_RE.match(basename)
    if not match:
        raise ValueError(
            f"Could not parse tile/channel from tile path '{tile_path}'. "
            "Expected a basename like tile_000000_ch_488(.ome).zarr or "
            "tile_x_0000_y_0000_z_0000_ch_488(.ome).zarr."
        )

    tile_name, channel, _suffix = match.groups()
    return tile_name, channel


def collect_tile_paths(tile_paths: list[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    tile_children: dict[str, dict[str, str]] = {}
    tile_order: list[str] = []

    for tile_path in tile_paths:
        tile_name, channel = parse_tile_path(tile_path)
        channels = tile_children.setdefault(tile_name, {})
        if not channels:
            tile_order.append(tile_name)
        if channel in channels:
            raise ValueError(
                f"Tile JSON contains duplicate paths for {tile_name} channel {channel}."
            )
        channels[channel] = tile_path

    return tile_children, tile_order


def safe_filename_component(value: str) -> str:
    safe_value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe_value or "tile"


def tile_output_prefix(tile_path: str) -> str:
    parts = [part for part in tile_path.rstrip("/").split("/") if part]
    tile_basename = parts[-1] if parts else "tile"
    parent_name = parts[-2] if len(parts) >= 2 else "tile"
    return (
        f"{safe_filename_component(parent_name)}_"
        f"{safe_filename_component(tile_basename)}"
    )


def resize_dask(image, scale_factor, order=1, output_chunks=(128, 256, 256)):
    """
    Resize a 3D Dask array using an affine transformation.

    This function scales a 3D Dask array by the given scale_factor along each dimension.
    It uses an affine transformation where the matrix maps output coordinates to input
    coordinates. For example, if scale_factor is 2, the output array will be twice as large
    in each dimension. When working with binary masks, consider using order=0 to preserve
    the binary nature.

    Parameters
    ----------
    image : dask.array
        The input 3D Dask array to be resized.
    scale_factor : float
        The scaling factor for each axis. For example, 2.0 will double the size.
    order : int, optional
        The order of the interpolation. Use order=0 for nearest-neighbor (good for binary
        masks), or higher orders for smoother results. Default is 1 (linear interpolation).
    output_chunks : tuple, optional
        The desired chunk size for the output Dask array. Default is (256, 256, 256).

    Returns
    -------
    dask.array
        The resized Dask array.
    """
    from dask_image.ndinterp import affine_transform

    # Construct a 4x4 homogeneous affine transformation matrix.
    # The matrix maps output coordinates into input coordinates.
    # Scaling factors are inverted because of this coordinate mapping.
    matrix = np.array([
        [1/scale_factor, 0, 0, 0],
        [0, 1/scale_factor, 0, 0],
        [0, 0, 1/scale_factor, 0],
        [0, 0, 0, 1]
    ])

    # Calculate the new output shape (assumes image has at least 3 dimensions).
    new_shape = tuple(int(dim * scale_factor) for dim in image.shape[:3])

    # Apply the affine transformation.
    resized_image = affine_transform(
        image,
        matrix=matrix,
        order=order,
        output_shape=new_shape,
        output_chunks=output_chunks
    )

    return resized_image


def read_level_grid(tile_group, level) -> dict:
    """Return the ZYX ``scale`` and ``translation`` of one pyramid level.

    Reads the OME-NGFF multiscales metadata of a tile group: ``attributes.ome``
    for Zarr v3 (NGFF 0.5) or the top-level attributes for Zarr v2 (NGFF 0.4).
    A level without a translation is placed at the origin.
    """
    attrs = dict(tile_group.attrs)
    multiscales = attrs.get("ome", attrs).get("multiscales")
    if not multiscales:
        raise ValueError("Tile group has no OME-NGFF multiscales metadata.")

    for dataset in multiscales[0]["datasets"]:
        if str(dataset["path"]) != str(level):
            continue
        scale = None
        translation = None
        for transform in dataset.get("coordinateTransformations", []):
            if transform.get("type") == "scale":
                scale = [float(v) for v in transform["scale"][-3:]]
            elif transform.get("type") == "translation":
                translation = [float(v) for v in transform["translation"][-3:]]
        if scale is None:
            raise ValueError(f"Level {level} has no scale transformation.")
        return {"scale": scale, "translation": translation or [0.0, 0.0, 0.0]}

    raise ValueError(f"Level {level} is not listed in the multiscales metadata.")


def physical_index_map(
    target_size: int,
    target_scale: float,
    target_translation: float,
    source_size: int,
    source_scale: float,
    source_translation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each target voxel along one axis to the nearest source voxel.

    Voxel centers are at ``translation + scale * index``, as in OME-NGFF. Returns
    the source index of every target voxel and whether that voxel lies inside the
    source's extent (within half a source voxel of its first or last center).
    """
    centers = target_translation + target_scale * np.arange(target_size)
    position = (centers - source_translation) / source_scale
    inside = (position >= -0.5) & (position <= source_size - 0.5)
    index = np.clip(np.rint(position).astype(np.int64), 0, source_size - 1)
    return index, inside


def resample_mask_to_grid(
    mask: np.ndarray,
    source_grid: dict,
    target_shape: tuple[int, int, int],
    target_grid: dict,
) -> np.ndarray:
    """Resample a ZYX mask onto another grid by physical position.

    Use this when two channels of a tile are sampled differently, for example a
    561 channel acquired at an 8 um z step and a 488 channel at 2 um. A plain
    resize by shape would stretch the mask whenever the channels cover different
    z extents. Target voxels outside the source's extent are ``False``.
    """
    if mask.ndim != 3 or len(target_shape) != 3:
        raise ValueError(
            f"Expected 3D (z, y, x) shapes, got {mask.shape} and {target_shape}."
        )
    result = np.asarray(mask, dtype=bool)
    inside_by_axis = []
    for axis in range(3):
        index, inside = physical_index_map(
            target_shape[axis],
            target_grid["scale"][axis],
            target_grid["translation"][axis],
            mask.shape[axis],
            source_grid["scale"][axis],
            source_grid["translation"][axis],
        )
        result = np.take(result, index, axis=axis)
        inside_by_axis.append(inside)
    inside_z, inside_y, inside_x = inside_by_axis
    return (
        result
        & inside_z[:, None, None]
        & inside_y[None, :, None]
        & inside_x[None, None, :]
    )
