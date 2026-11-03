import numpy as np
import pytest

from lesionbench.metrics.froc import (
    detections_from_kept,
    evaluation_mask,
    froc_curve,
    froc_score,
    slide_probabilities,
    to_level,
)


def test_evaluation_mask_merges_within_75um_and_flags_itcs():
    # 10 µm per pixel: dilation keeps distance < 3.75 px; ITC if major axis < 27.5 px
    m = np.ones((100, 100), np.uint8)  # 1 = normal tissue, 2 = tumour
    m[10:13, 10:13] = 2  # small lesion ...
    m[10:13, 19:22] = 2  # ... 6 px (60 µm) away: merges with it
    m[50:53, 10:13] = 2  # 9 px (90 µm) from the next: stays apart
    m[50:53, 22:25] = 2
    m[80:85, 20:70] = 2  # 50 px long: not an ITC
    labels, itc = evaluation_mask(m, level_spacing_um=10.0)
    assert labels.max() == 4
    assert labels[11, 11] == labels[11, 20] != 0
    assert labels[51, 11] != labels[51, 23]
    big = labels[82, 40]
    assert big not in itc and {labels[11, 11], labels[51, 11], labels[51, 23]} == itc
    assert evaluation_mask(m, 10.0, include_itcs=True)[1] == set()


def test_evaluation_mask_fills_holes():
    m = np.ones((60, 60), np.uint8)
    m[10:50, 10:50] = 2
    m[20:40, 20:40] = 1  # a 20 px hole, wider than the dilation closes
    labels, _ = evaluation_mask(m, 10.0)
    assert labels[30, 30] == labels[15, 15] == 1


def test_to_level_rounds_level0_xy_to_row_col():
    assert to_level([(0.9, 100, 50)], 32.0) == [(0.9, 2, 3)]


def test_slide_probabilities_one_hit_per_region_itcs_ignored():
    mask = np.zeros((10, 10), int)
    mask[0:3, 0:3] = 1
    mask[6:9, 6:9] = 2
    dets = [(0.3, 1, 1), (0.7, 2, 2), (0.9, 7, 7), (0.5, 5, 0)]
    fp, tp, n = slide_probabilities(dets, mask, itc_labels={2})
    assert fp == [0.5] and tp == [0.7, 0.0] and n == 1


def test_slide_probabilities_normal_slide_all_false_positives():
    assert slide_probabilities([(0.4, 0, 0), (0.6, 3, 3)], None, set()) == ([0.4, 0.6], [], 0)


def test_froc_curve_and_score_by_hand():
    tumour = ([0.5], [0.7, 0.4], 2)  # one FP, two regions hit at 0.7 and 0.4
    normal = ([0.6, 0.2], [], 0)
    fps, sens = froc_curve([tumour, normal])
    assert fps == [1.5, 1.0, 1.0, 0.5, 0.0, 0.0]  # thresholds 0.2, 0.4, 0.5, 0.6, 0.7, then none
    assert sens == [1.0, 1.0, 0.5, 0.5, 0.5, 0.0]
    # step lookup: 1/4, 1/2 and 1 FP read 0.5; 2, 4 and 8 FP run past the end and read 1.0
    assert froc_score(fps, sens) == pytest.approx(0.75)


def test_froc_curve_ignores_missed_regions_as_thresholds():
    fps, sens = froc_curve([([], [0.8, 0.0], 2)])
    assert fps == [0.0, 0.0] and sens == [0.5, 0.0]


def test_detections_are_local_maxima_of_kept_patches():
    ps = 10
    coords = np.array([[0, 0], [10, 10], [50, 0], [20, 0], [0, 30], [10, 30], [20, 30]])
    probs = np.array([0.2, 0.8, 0.5, 0.99, 0.9, 0.1, 0.7])
    # row 0 is beaten by its kept neighbour row 1 (row 3 is higher but not kept); row 2 is isolated;
    # rows 4-6 are a chain whose two ends are both peaks (they are not neighbours)
    dets = detections_from_kept([0, 1, 2, 4, 5, 6], coords, ps, probs)
    assert sorted(dets) == [(0.5, 55, 5), (0.7, 25, 35), (0.8, 15, 15), (0.9, 5, 35)]
    assert detections_from_kept([], coords, ps, probs) == []


def test_detections_tie_goes_to_lower_row_and_full_bag_gives_peaks():
    coords = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]])
    assert detections_from_kept([1, 0], coords, 10, np.array([0.5, 0.5, 0, 0, 0])) == [(0.5, 5, 5)]
    probs = np.array([0.1, 0.6, 0.3, 0.4, 0.2])  # every patch kept: peaks at rows 1 and 3
    assert detections_from_kept(np.arange(5), coords, 10, probs) == [(0.6, 15, 5), (0.4, 35, 5)]


def test_read_mask_level_spacing_and_downsample(tmp_path):
    import tifffile

    from lesionbench.eval.froc import read_mask

    base = np.zeros((64, 128), np.uint8)
    base[8:16, 8:24] = 2
    path = tmp_path / "m.tif"
    with tifffile.TiffWriter(path) as tw:  # 0.25 µm/px = 40,000 px/cm; one sub-resolution at /4
        tw.write(base, subifds=1, resolution=(40000, 40000), resolutionunit="CENTIMETER", tile=(16, 16))
        tw.write(base[::4, ::4], subfiletype=1, tile=(16, 16))
    mask, spacing, ds = read_mask(path, 1)
    assert mask.shape == (16, 32) and ds == 4.0 and spacing == pytest.approx(1.0)
    assert set(np.unique(mask)) == {0, 2}


def test_scorers_load_saved_weights_and_return_probabilities(tmp_path):
    import torch

    from lesionbench.eval.froc import load_scorer
    from lesionbench.selectors.abmil import GatedABMIL

    torch.manual_seed(0)
    m = GatedABMIL(8)
    (tmp_path / "abmil_full/fold0/Kfull").mkdir(parents=True)
    torch.save(m.state_dict(), tmp_path / "abmil_full/fold0/Kfull/model.pt")
    head = torch.nn.Linear(8, 2)
    (tmp_path / "pmil/fold0").mkdir(parents=True)
    torch.save(head.state_dict(), tmp_path / "pmil/fold0/maxmil_head.pt")
    x = torch.randn(5, 8)
    p = load_scorer("abmil", tmp_path, 0, 8)(x)
    m.eval()
    with torch.no_grad():
        expected = torch.softmax(m.classifier(m.fc(x)), dim=1)[:, 1].double().numpy()
        # linear classifier: slide logits = attention-weighted mean of the patch logits
        a = torch.softmax(m(x)[1], dim=0)
        torch.testing.assert_close(m(x)[0], a @ m.classifier(m.fc(x)))
    np.testing.assert_allclose(p, expected)
    q = load_scorer("maxmil", tmp_path, 0, 8)(x)
    assert q.shape == (5,) and ((q > 0) & (q < 1)).all()


def test_run_froc_end_to_end_on_toy_test_set(tmp_path, monkeypatch):
    import csv
    import importlib.util

    import h5py
    import torch

    from lesionbench.config import ROOT

    spec = importlib.util.spec_from_file_location("run_froc", ROOT / "scripts" / "run_froc.py")
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)

    ps = 512  # level-0 patch = 16 px of the level-5 mask (downsample 32)
    coords = np.array([[x * ps, y * ps] for y in range(4) for x in range(4)])
    (tmp_path / "cache/uni").mkdir(parents=True)
    for sid, n_les in (("test_000", 1), ("test_001", 0)):
        feats = np.zeros((16, 4), np.float32)
        feats[:, 0] = np.linspace(-2, 1, 16)
        if n_les:
            feats[5, 0] = 5.0  # the tumour patch, at grid (1, 1)
        with h5py.File(tmp_path / "cache/uni" / f"{sid}.h5", "w") as f:
            f["features"], f["coords"] = feats, coords
            f.attrs.update({"patch_size_level0": ps, "n_lesions": n_les})
    (tmp_path / "data/evaluation").mkdir(parents=True)
    (tmp_path / "data/evaluation/reference.csv").write_text(
        "image,type,class,center\ntest_000.tif,tumor,micro,0\ntest_001.tif,normal,negative,0\n")
    mask = np.ones((64, 64), np.uint8)
    mask[16:32, 16:32] = 2  # 16 px x 20 µm = 320 µm: not an ITC
    monkeypatch.setattr(rf, "read_mask", lambda path, level: (mask, 20.0, 32.0))
    monkeypatch.setattr(rf, "SCORERS", ("toy",))
    monkeypatch.setattr(rf, "load_scorer", lambda *a: lambda x: torch.sigmoid(x[:, 0]).double().numpy())
    runs = tmp_path / "runs/uni"
    for method, kept in {"hit": {"test_000": [5, 15], "test_001": [3]},
                         "miss": {"test_000": [0, 15], "test_001": [3]}}.items():
        (runs / method / "fold0/K2").mkdir(parents=True)
        np.savez(runs / method / "fold0/K2/retained.npz",
                 **{f"retained/{s}": np.array(v) for s, v in kept.items()})
    cfg = {"paths": {"data_root": str(tmp_path / "data"), "cache_root": str(tmp_path / "cache")},
           "training": {"n_folds": 1, "dropout": 0.25}}
    rf.froc(cfg, "uni", runs, tmp_path / "froc")
    rows = {(r["method"], r["itc"]): r for r in csv.DictReader(open(tmp_path / "froc/toy/runs.csv"))}
    assert float(rows[("hit", "0")]["froc"]) == 1.0 and float(rows[("miss", "0")]["froc"]) == 0.0
    # hit keeps rows 5 and 15 on the tumour slide (not neighbours: two detections) and one on the normal slide
    assert float(rows[("hit", "0")]["detections_per_slide"]) == 1.5
    assert (tmp_path / "froc/toy/summary.csv").exists() and (tmp_path / "froc/run.json").exists()
