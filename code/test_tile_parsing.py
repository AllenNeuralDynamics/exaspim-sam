#!/usr/bin/env python3
"""Unit tests for tile-name parsing across both exaSPIM naming schemes.

Covers the flat index scheme (``tile_000000_ch_488.zarr``) and the
grid-coordinate scheme (``tile_x_0000_y_0000_z_0000_ch_488.zarr``).

Runs standalone (``python3 code/test_tile_parsing.py``) and is also
pytest-compatible (``python3 -m pytest code/test_tile_parsing.py``). The core
tests only need ``numpy`` (pulled in by ``utils``); the optional
``postprocess_masks`` check is skipped if its image dependencies are absent.
"""

from utils import (
    TILE_NAME_SEARCH_RE,
    TILE_PATH_RE,
    collect_tile_paths,
    parse_tile_path,
)

# postprocess_masks pulls in tifffile/scipy/skimage; only test its wrapper when
# those are importable so this file still runs in a bare numpy environment.
try:
    from postprocess_masks import detect_tile_name_from_filename

    _HAVE_POSTPROCESS = True
except Exception:  # pragma: no cover - depends on optional image deps
    _HAVE_POSTPROCESS = False

GRID = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38/SPIM.ome.zarr"
FLAT = "s3://aind-open-data/exaSPIM_754615_2025-01-23_16-44-53/SPIM.ome.zarr"


def test_parse_grid_scheme():
    assert parse_tile_path(f"{GRID}/tile_x_0000_y_0000_z_0000_ch_488.zarr") == (
        "tile_x_0000_y_0000_z_0000",
        "488",
    )
    # Grid names also appear with multi-digit indices and a trailing slash.
    assert parse_tile_path(f"{GRID}/tile_x_0012_y_0003_z_0001_ch_561.zarr/") == (
        "tile_x_0012_y_0003_z_0001",
        "561",
    )


def test_parse_flat_scheme_still_works():
    assert parse_tile_path(f"{FLAT}/tile_000000_ch_488.zarr") == (
        "tile_000000",
        "488",
    )
    # Zarr v3 ``.ome.zarr`` suffix stays supported.
    assert parse_tile_path(f"{FLAT}/tile_000006_ch_561.ome.zarr") == (
        "tile_000006",
        "561",
    )


def test_parse_rejects_bad_names():
    bad = [
        f"{GRID}/tile_x_0000_ch_488.zarr",          # incomplete grid (no y/z)
        f"{GRID}/some_random_group_ch_488.zarr",     # not a tile
        f"{FLAT}/tile_000000_ch_405.zarr",           # channel outside 488/561
        f"{GRID}/.zgroup",                            # zarr metadata, not a tile
    ]
    for path in bad:
        try:
            parse_tile_path(path)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {path!r}")


def test_tile_path_re_anchoring():
    assert TILE_PATH_RE.match("tile_x_0000_y_0000_z_0000_ch_488.zarr")
    assert TILE_PATH_RE.match("tile_000000_ch_561.ome.zarr")
    assert TILE_PATH_RE.match("median_summary") is None


def test_collect_pairs_grid_tile_across_channels():
    paths = [
        f"{GRID}/tile_x_0001_y_0002_z_0000_ch_488.zarr",
        f"{GRID}/tile_x_0001_y_0002_z_0000_ch_561.zarr",
    ]
    tile_children, tile_order = collect_tile_paths(paths)
    assert tile_order == ["tile_x_0001_y_0002_z_0000"]
    assert set(tile_children["tile_x_0001_y_0002_z_0000"]) == {"488", "561"}


def test_search_re_finds_token_in_output_filename():
    grid_fn = (
        "exaSPIM_708373_2024-04-02_19-49-38_"
        "tile_x_0000_y_0000_z_0000_ch_488.zarr_data.tif"
    )
    flat_fn = "exaSPIM_754615_2025-01-23_16-44-53_tile_000000_ch_488.zarr_pred.tif"
    grid_match = TILE_NAME_SEARCH_RE.search(grid_fn)
    flat_match = TILE_NAME_SEARCH_RE.search(flat_fn)
    assert grid_match and grid_match.group(1) == "tile_x_0000_y_0000_z_0000"
    assert flat_match and flat_match.group(1) == "tile_000000"
    assert TILE_NAME_SEARCH_RE.search("no_tile_token_here.tif") is None


def test_postprocess_detect_tile_name():
    if not _HAVE_POSTPROCESS:
        print("  (skipped: postprocess_masks image deps unavailable)")
        return
    grid_fn = (
        "exaSPIM_708373_2024-04-02_19-49-38_"
        "tile_x_0000_y_0000_z_0000_ch_488.zarr_data.tif"
    )
    assert (
        detect_tile_name_from_filename(grid_fn) == "tile_x_0000_y_0000_z_0000"
    )
    assert detect_tile_name_from_filename("no_tile.tif") is None


def _main() -> int:
    tests = [
        test_parse_grid_scheme,
        test_parse_flat_scheme_still_works,
        test_parse_rejects_bad_names,
        test_tile_path_re_anchoring,
        test_collect_pairs_grid_tile_across_channels,
        test_search_re_finds_token_in_output_filename,
        test_postprocess_detect_tile_name,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - surface any failure
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
