import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from lesionbench.features.encoders import load_encoder
from lesionbench.features.extract import PatchDataset, extract_features, read_patch, write_feature_h5
from lesionbench.preprocess.pipeline import write_patch_h5


class FakeSlide:
    """OpenSlide-like pyramid (downsamples 1, 2, 4) over a level-0 RGB array."""

    level_downsamples = (1.0, 2.0, 4.0)

    def __init__(self, level0: np.ndarray):
        self.levels = [level0[::d, ::d] for d in (1, 2, 4)]

    def get_best_level_for_downsample(self, ds):
        return max(i for i, d in enumerate(self.level_downsamples) if d <= ds)

    def read_region(self, loc, level, size):
        d = int(self.level_downsamples[level])
        x, y = loc[0] // d, loc[1] // d
        rgb = self.levels[level][y : y + size[1], x : x + size[0]]
        return Image.fromarray(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]), "RGBA")


def _slide():
    # 32 x 32 level-0 image; each 8 x 8 block has its own colour (block index in the red channel).
    img = np.zeros((32, 32, 3), np.uint8)
    for by in range(4):
        for bx in range(4):
            img[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8] = (by * 4 + bx, 100, 200)
    return FakeSlide(img)


def test_read_patch_uses_coarser_level_and_resizes():
    s = _slide()
    p = read_patch(s, 8, 16, patch_size_l0=8, out_size=4)  # ds 2 -> level 1, 4 px, no resize needed
    assert p.size == (4, 4) and p.mode == "RGB"
    assert (np.asarray(p)[..., 0] == 2 * 4 + 1).all()  # block (bx=1, by=2)
    p = read_patch(s, 8, 16, patch_size_l0=8, out_size=8)  # ds 1 -> level 0, exact
    assert (np.asarray(p)[..., 0] == 9).all()
    p = read_patch(s, 0, 0, patch_size_l0=16, out_size=6)  # ds 2.67 -> level 1, 8 px resized to 6
    assert p.size == (6, 6)


def test_extract_features_keeps_row_order_across_batches():
    s = _slide()
    coords = np.array([[x, y] for y in range(0, 32, 8) for x in range(0, 32, 8)])[::-1]  # reversed order
    embed, tf, dim = load_encoder("dummy")
    ds = PatchDataset("fake", coords, patch_size_l0=8, out_size=4, transform=tf, opener=lambda _: s)
    feats = extract_features(ds, embed, batch_size=3)
    assert feats.shape == (16, dim) and feats.dtype == np.float32
    red = feats[:, 0] * 255  # channel-0 mean, back on the 0-255 scale
    np.testing.assert_allclose(red, [(y // 8) * 4 + x // 8 for x, y in coords], atol=1e-3)


def test_write_feature_h5_schema_and_checks(tmp_path):
    res = {"coords": np.array([[0, 0], [8, 0]]), "patch_labels": np.array([0, 1]),
           "lesion_ids": np.array([-1, 0]), "attrs": {"slide_id": "s", "stratum": "itc"}}
    write_patch_h5(tmp_path / "patches" / "s.h5", res)
    out = tmp_path / "uni" / "s.h5"
    write_feature_h5(out, np.ones((2, 4)), tmp_path / "patches" / "s.h5", "uni", git_sha="abc")
    with h5py.File(out) as f:
        assert set(f) == {"features", "coords", "patch_labels", "lesion_ids"}
        assert f["features"].dtype == np.float32 and f["features"].shape == (2, 4)
        assert f["coords"][:].tolist() == [[0, 0], [8, 0]] and f["lesion_ids"][:].tolist() == [-1, 0]
        assert f.attrs["backbone"] == "uni" and f.attrs["slide_id"] == "s" and f.attrs["feature_dim"] == 4
    assert not list(out.parent.glob("*.tmp"))
    with pytest.raises(ValueError, match="3 feature rows for 2 patches"):
        write_feature_h5(tmp_path / "x.h5", np.ones((3, 4)), tmp_path / "patches" / "s.h5", "uni")


def test_load_encoder_unknown_name():
    with pytest.raises(ValueError):
        load_encoder("nope")
    embed, tf, dim = load_encoder("dummy")
    assert embed(torch.zeros(2, 3, 4, 4)).shape == (2, dim)
