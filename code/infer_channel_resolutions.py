#!/usr/bin/env python3
"""Infer channel resolution levels from paired tile array sizes."""

from __future__ import annotations

import argparse
import math
import sys

import zarr
from utils import TILE_PATH_RE, collect_tile_paths, load_tile_paths


CHANNELS = ("488", "561")
LARGER_CHANNEL_RES = 4
SMALLER_CHANNEL_RES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer 488/561 resolution levels by comparing level-0 array sizes "
            "for the first paired tile in a parent SPIM.ome.zarr group or "
            "tile_paths JSON file. "
            "Equal channel sizes use resolution level 4 for both channels."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zarr", help="Parent SPIM.ome.zarr path.")
    source.add_argument(
        "--tile-json",
        help="JSON file containing a tile_paths list of tile zarr paths.",
    )
    return parser.parse_args()


def collect_tile_children(zarr_group: zarr.Group) -> dict[str, dict[str, str]]:
    """Map each tile name to its {channel: actual child group name}.

    The child name is captured verbatim so the level-0 shape lookup can address
    both the v2 (``tile_..._ch_488.zarr``) and v3 (``tile_..._ch_488.ome.zarr``)
    naming without reconstructing the suffix.
    """
    tile_children: dict[str, dict[str, str]] = {}

    for group_name in zarr_group.keys():
        match = TILE_PATH_RE.match(group_name)
        if not match:
            continue

        tile_name, channel, _suffix = match.groups()
        tile_children.setdefault(tile_name, {})[channel] = group_name

    return tile_children


def paired_tiles(tile_children: dict[str, dict[str, str]]) -> list[str]:
    return sorted(
        tile_name
        for tile_name, channels in tile_children.items()
        if all(channel in channels for channel in CHANNELS)
    )


def paired_tiles_in_order(
    tile_children: dict[str, dict[str, str]], tile_order: list[str]
) -> list[str]:
    return [
        tile_name
        for tile_name in tile_order
        if all(channel in tile_children[tile_name] for channel in CHANNELS)
    ]


def get_level_zero_shape(
    zarr_group: zarr.Group, group_name: str
) -> tuple[int, ...]:
    try:
        return tuple(int(axis) for axis in zarr_group[group_name]["0"].shape)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read level 0 shape for {group_name}: {exc}"
        ) from exc


def get_level_zero_shape_from_path(tile_path: str) -> tuple[int, ...]:
    try:
        tile_group = zarr.open_group(tile_path, mode="r")
        return tuple(int(axis) for axis in tile_group["0"].shape)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read level 0 shape for {tile_path}: {exc}"
        ) from exc


def infer_resolutions_from_shapes(
    shape_488: tuple[int, ...], shape_561: tuple[int, ...], reference_tile: str
) -> tuple[int, int, str]:
    size_488 = math.prod(shape_488)
    size_561 = math.prod(shape_561)

    if size_488 == size_561:
        return LARGER_CHANNEL_RES, LARGER_CHANNEL_RES, reference_tile

    if size_488 > size_561:
        return LARGER_CHANNEL_RES, SMALLER_CHANNEL_RES, reference_tile

    return SMALLER_CHANNEL_RES, LARGER_CHANNEL_RES, reference_tile


def infer_resolutions(zarr_path: str) -> tuple[int, int, str]:
    try:
        zarr_group = zarr.open_group(zarr_path, mode="r")
    except Exception as exc:
        raise RuntimeError(f"Could not open zarr group {zarr_path}: {exc}") from exc

    tile_children = collect_tile_children(zarr_group)
    tiles = paired_tiles(tile_children)
    if not tiles:
        raise RuntimeError(
            "No paired tile groups found. Expected parent group entries like "
            "tile_000000_ch_488(.ome).zarr and tile_000000_ch_561(.ome).zarr, "
            "or the grid form tile_x_0000_y_0000_z_0000_ch_488(.ome).zarr and "
            "tile_x_0000_y_0000_z_0000_ch_561(.ome).zarr."
        )

    reference_tile = tiles[0]
    shape_488 = get_level_zero_shape(zarr_group, tile_children[reference_tile]["488"])
    shape_561 = get_level_zero_shape(zarr_group, tile_children[reference_tile]["561"])
    return infer_resolutions_from_shapes(shape_488, shape_561, reference_tile)


def infer_resolutions_from_tile_json(tile_json_path: str) -> tuple[int, int, str]:
    tile_paths = load_tile_paths(tile_json_path)
    tile_children, tile_order = collect_tile_paths(tile_paths)
    tiles = paired_tiles_in_order(tile_children, tile_order)
    if not tiles:
        raise RuntimeError(
            f"No paired tile_paths found in {tile_json_path}. Expected entries for "
            "both channels of at least one tile, e.g. tile_000000_ch_488(.ome).zarr "
            "and tile_000000_ch_561(.ome).zarr, or the grid form "
            "tile_x_0000_y_0000_z_0000_ch_488(.ome).zarr and its ch_561 counterpart."
        )

    reference_tile = tiles[0]
    shape_488 = get_level_zero_shape_from_path(tile_children[reference_tile]["488"])
    shape_561 = get_level_zero_shape_from_path(tile_children[reference_tile]["561"])
    return infer_resolutions_from_shapes(shape_488, shape_561, reference_tile)


def main() -> int:
    args = parse_args()

    try:
        if args.tile_json:
            res_488, res_561, reference_tile = infer_resolutions_from_tile_json(
                args.tile_json
            )
        else:
            res_488, res_561, reference_tile = infer_resolutions(args.zarr)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"RES_488={res_488}")
    print(f"RES_561={res_561}")
    print(f"REFERENCE_TILE={reference_tile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
