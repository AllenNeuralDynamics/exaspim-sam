#!/usr/bin/env python3
"""Infer channel resolution levels from paired tile array sizes."""

from __future__ import annotations

import argparse
import math
import re
import sys

import zarr


CHANNELS = ("488", "561")
TILE_GROUP_RE = re.compile(r"^(tile_\d+)_ch_(488|561)\.zarr$")
LARGER_CHANNEL_RES = 4
SMALLER_CHANNEL_RES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer 488/561 resolution levels by comparing level-0 array sizes "
            "for the first paired tile in a parent SPIM.ome.zarr group. "
            "Equal channel sizes use resolution level 4 for both channels."
        )
    )
    parser.add_argument("--zarr", required=True, help="Parent SPIM.ome.zarr path.")
    return parser.parse_args()


def collect_paired_tiles(zarr_group: zarr.hierarchy.Group) -> list[str]:
    tile_channels: dict[str, set[str]] = {}

    for group_name in zarr_group.keys():
        match = TILE_GROUP_RE.match(group_name)
        if not match:
            continue

        tile_name, channel = match.groups()
        tile_channels.setdefault(tile_name, set()).add(channel)

    return sorted(
        tile_name
        for tile_name, channels in tile_channels.items()
        if all(channel in channels for channel in CHANNELS)
    )


def get_level_zero_shape(
    zarr_group: zarr.hierarchy.Group, tile_name: str, channel: str
) -> tuple[int, ...]:
    group_name = f"{tile_name}_ch_{channel}.zarr"
    try:
        return tuple(int(axis) for axis in zarr_group[group_name]["0"].shape)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read level 0 shape for {group_name}: {exc}"
        ) from exc


def infer_resolutions(zarr_path: str) -> tuple[int, int, str]:
    try:
        zarr_group = zarr.open_group(zarr_path, mode="r")
    except Exception as exc:
        raise RuntimeError(f"Could not open zarr group {zarr_path}: {exc}") from exc

    paired_tiles = collect_paired_tiles(zarr_group)
    if not paired_tiles:
        raise RuntimeError(
            "No paired tile groups found. Expected parent group entries like "
            "tile_000000_ch_488.zarr and tile_000000_ch_561.zarr."
        )

    reference_tile = paired_tiles[0]
    shape_488 = get_level_zero_shape(zarr_group, reference_tile, "488")
    shape_561 = get_level_zero_shape(zarr_group, reference_tile, "561")
    size_488 = math.prod(shape_488)
    size_561 = math.prod(shape_561)

    if size_488 == size_561:
        return LARGER_CHANNEL_RES, LARGER_CHANNEL_RES, reference_tile

    if size_488 > size_561:
        return LARGER_CHANNEL_RES, SMALLER_CHANNEL_RES, reference_tile

    return SMALLER_CHANNEL_RES, LARGER_CHANNEL_RES, reference_tile


def main() -> int:
    args = parse_args()

    try:
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
