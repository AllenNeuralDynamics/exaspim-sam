#!/usr/bin/env python3
"""Tests for channels that are binned or sampled at different z steps.

Run with ``python3 -m pytest code`` (needs numpy, zarr, tifffile, scipy,
scikit-image and tqdm).
"""

import json

import numpy as np
import tifffile
import zarr

from infer_channel_resolutions import infer_resolutions_from_shapes
from postprocess_masks import post_process_mask
from utils import physical_index_map, read_level_grid, resample_mask_to_grid

# Level-0 shapes (t, c, z, y, x) of real exaSPIM datasets.
# 859684: 561 binned 8x in z, y and x.
FULL_859684 = (1, 1, 20480, 10640, 14192)
BINNED_859684 = (1, 1, 2560, 1330, 1774)
# 853730: 561 at an 8 um z step, 488 at 2 um, same 0.748 um pixel. Not binned.
FULL_853730 = (1, 1, 12288, 10640, 14192)
COARSE_Z_853730 = (1, 1, 2944, 10640, 14192)


def test_binned_561_reads_level_1():
    assert infer_resolutions_from_shapes(FULL_859684, BINNED_859684, "t") == (4, 1, "t")


def test_binned_488_reads_level_1():
    assert infer_resolutions_from_shapes(BINNED_859684, FULL_859684, "t") == (1, 4, "t")


def test_coarser_z_step_reads_level_4_for_both():
    assert infer_resolutions_from_shapes(FULL_853730, COARSE_Z_853730, "t") == (4, 4, "t")
    assert infer_resolutions_from_shapes(COARSE_Z_853730, FULL_853730, "t") == (4, 4, "t")


def test_equal_shapes_read_level_4_for_both():
    assert infer_resolutions_from_shapes(FULL_853730, FULL_853730, "t") == (4, 4, "t")


def _datasets(scale_z):
    return [
        {
            "path": str(level),
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, scale_z * 2**level, 0.748 * 2**level, 0.748 * 2**level]},
                {"type": "translation", "translation": [0.0, 0.0, 100.0 + level, 5.0, -7.0]},
            ],
        }
        for level in range(5)
    ]


def test_read_level_grid_zarr_v3(tmp_path):
    group = zarr.open_group(str(tmp_path / "tile.ome.zarr"), mode="w", zarr_format=3)
    group.attrs["ome"] = {"multiscales": [{"datasets": _datasets(8.0)}]}
    grid = read_level_grid(zarr.open_group(str(tmp_path / "tile.ome.zarr"), mode="r"), 4)
    assert grid == {"scale": [128.0, 11.968, 11.968], "translation": [104.0, 5.0, -7.0]}


def test_read_level_grid_zarr_v2(tmp_path):
    group = zarr.open_group(str(tmp_path / "tile.zarr"), mode="w", zarr_format=2)
    group.attrs["multiscales"] = [{"datasets": _datasets(2.0)}]
    grid = read_level_grid(zarr.open_group(str(tmp_path / "tile.zarr"), mode="r"), "1")
    assert grid == {"scale": [4.0, 1.496, 1.496], "translation": [101.0, 5.0, -7.0]}


def test_read_level_grid_missing_level_raises(tmp_path):
    group = zarr.open_group(str(tmp_path / "tile.ome.zarr"), mode="w", zarr_format=3)
    group.attrs["ome"] = {"multiscales": [{"datasets": _datasets(8.0)}]}
    try:
        read_level_grid(group, 7)
    except ValueError as exc:
        assert "Level 7" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# Level-4 grids of tile 0 in 853730 along z: 488 at 32 um, 561 at 128 um. The 561
# stack ends about 1 mm shallower than the 488 stack.
GRID_488 = {"scale": [32.0, 12.0, 12.0], "translation": [24071.5, 0.0, 0.0]}
GRID_561 = {"scale": [128.0, 12.0, 12.0], "translation": [24116.5, 0.0, 0.0]}


def test_physical_index_map_coarser_target():
    index, inside = physical_index_map(5, 128.0, 24116.5, 24, 32.0, 24071.5)
    # 561 slice k sits at 488 slice 1.41 + 4k.
    assert index.tolist() == [1, 5, 9, 13, 17]
    assert inside.all()


def test_physical_index_map_marks_voxels_beyond_the_source():
    index, inside = physical_index_map(24, 32.0, 24071.5, 5, 128.0, 24116.5)
    # The 561 stack covers 488 slices 0..19; 20..23 are deeper than it.
    assert inside.tolist() == [True] * 20 + [False] * 4
    assert index[:20].tolist() == [round(((24071.5 + 32 * i) - 24116.5) / 128) for i in range(20)]


def test_resample_mask_488_onto_561():
    mask_488 = np.zeros((24, 3, 4), dtype=bool)
    mask_488[4:12] = True
    mask_561 = resample_mask_to_grid(mask_488, GRID_488, (5, 3, 4), GRID_561)
    assert mask_561.shape == (5, 3, 4)
    assert mask_561[:, 0, 0].tolist() == [False, True, True, False, False]
    assert (mask_561 == mask_561[:, :1, :1]).all()


def test_resample_mask_561_onto_488_is_false_beyond_561():
    mask_561 = np.ones((5, 3, 4), dtype=bool)
    mask_488 = resample_mask_to_grid(mask_561, GRID_561, (24, 3, 4), GRID_488)
    assert mask_488[:, 1, 1].tolist() == [True] * 20 + [False] * 4


def test_resample_same_grid_is_identity():
    rng = np.random.default_rng(0)
    mask = rng.random((6, 5, 4)) > 0.5
    assert (resample_mask_to_grid(mask, GRID_488, mask.shape, GRID_488) == mask).all()


def _write_tile(tmp_path, inference_channel, write_grids=True):
    """Write the files inference_preprocess and inference leave for one tile."""
    prefix = "SPIM.ome.zarr_tile_000000_ch_{}.ome.zarr"
    raw_dir = tmp_path / "raw"
    # Big enough in YX for the 10000-voxel object filter on either grid.
    shapes = {"488": (24, 60, 60), "561": (5, 60, 60)}
    grids = {"488": GRID_488, "561": GRID_561}
    values = {"488": 100, "561": 300}
    for channel, shape in shapes.items():
        (raw_dir / channel).mkdir(parents=True)
        raw = np.full(shape, values[channel], dtype=np.uint16)
        tifffile.imwrite(raw_dir / channel / f"{prefix.format(channel)}_data.tif", raw)
        if write_grids:
            with open(raw_dir / channel / f"{prefix.format(channel)}_grid.json", "w") as f:
                json.dump({"shape": list(shape), **grids[channel]}, f)

    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    pred = np.zeros(shapes[inference_channel], dtype=np.uint8)
    pred[1:-1] = 1
    pred_path = pred_dir / f"{prefix.format(inference_channel)}_pred.tif"
    tifffile.imwrite(pred_path, pred)
    return pred_path, raw_dir


def test_postprocess_summarizes_both_channels_when_z_steps_differ(tmp_path):
    for inference_channel in ("488", "561"):
        case = tmp_path / inference_channel
        pred_path, raw_dir = _write_tile(case, inference_channel)
        medians = {}
        output = case / "out.tif"
        assert post_process_mask(str(pred_path), str(output), 0, 75.0, str(raw_dir), medians)
        assert medians == {"488": [100.0], "561": [300.0]}
        # The saved mask stays on the inference channel's grid.
        assert tifffile.imread(output).shape == tifffile.imread(pred_path).shape


def test_postprocess_fails_without_grids_when_shapes_differ(tmp_path):
    pred_path, raw_dir = _write_tile(tmp_path, "488", write_grids=False)
    output = tmp_path / "out.tif"
    assert not post_process_mask(str(pred_path), str(output), 0, 75.0, str(raw_dir), {})
    assert not output.exists()


def test_postprocess_writes_no_mask_for_a_tile_without_tissue(tmp_path):
    pred_path, raw_dir = _write_tile(tmp_path, "488")
    tifffile.imwrite(pred_path, np.zeros((24, 60, 60), dtype=np.uint8))
    output = tmp_path / "out.tif"
    medians = {}
    assert post_process_mask(str(pred_path), str(output), 0, 75.0, str(raw_dir), medians)
    assert not output.exists()
    assert medians == {}
