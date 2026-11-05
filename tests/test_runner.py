import csv
import json

import h5py
import numpy as np
import pytest

from lesionbench.eval import methods
from lesionbench.eval.data import load_slide, slide_ids
from lesionbench.eval.methods import METHODS
from lesionbench.eval.runner import fold_split, run_method_fold

CFG = {"seed": 0, "budgets": [4], "lsr_overlap_thresholds": [0.0, 0.1], "primary_overlap_threshold": 0.05,
       "strict_overlap_threshold": 0.6,
       "training": {"n_folds": 2, "lr": 1e-3, "weight_decay": 1e-5, "max_epochs": 2, "patience": 2,
                    "stop_epoch": 2, "dropout": 0.25, "epoch_cap": 2}}
PS = 512  # patch size at level 0


def _write_slide(root, sid, tumour, g):
    """6 x 5 grid of patches in two tissue types (so PMIL finds > 1 prototype); tumour slides carry one 2 x 2
    lesion (+4 shift) covering 4 patches."""
    coords = np.array([[x * PS, y * PS] for y in range(5) for x in range(6)])
    feats = g.normal(size=(30, 8)).astype(np.float32)
    feats[15:, 1] += 8.0
    labels, ids = np.zeros(30, np.uint8), np.full(30, -1, np.int32)
    rows = np.array([7, 8, 13, 14]) if tumour else np.array([], int)
    feats[rows] += 4.0
    labels[rows], ids[rows] = 1, 0
    attrs = {"slide_id": sid, "n_lesions": int(tumour), "stratum": "micro" if tumour else "negative",
             "patch_size_level0": PS}
    with h5py.File(root / "patches" / f"{sid}.h5", "w") as f:
        f.create_dataset("coords", data=coords)
        grp = f.create_group("overlap")
        grp.create_dataset("rows", data=rows.astype(np.int64))
        grp.create_dataset("lesion_ids", data=np.zeros(len(rows), np.int32))
        grp.create_dataset("fracs", data=np.full(len(rows), 0.5, np.float32))
        grp.create_dataset("lesion_diameter_um", data=np.full(int(tumour), 300.0, np.float32))
        f.attrs.update(attrs)
    with h5py.File(root / "uni" / f"{sid}.h5", "w") as f:
        for k, v in {"features": feats, "coords": coords, "patch_labels": labels, "lesion_ids": ids}.items():
            f.create_dataset(k, data=v)
        f.attrs.update(attrs)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache")
    (root / "patches").mkdir()
    (root / "uni").mkdir()
    g = np.random.default_rng(0)
    for i in range(6):
        _write_slide(root, f"tumor_{i:03d}", True, g)
        _write_slide(root, f"normal_{i:03d}", False, g)
    for i in range(4):
        _write_slide(root, f"test_{i:03d}", i % 2 == 0, g)
    return root


def _slides(cache):
    return [load_slide(cache, "uni", s) for s in slide_ids(cache, "uni")]


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_load_slide_labels_split_and_grid(cache):
    slides = {s.slide_id: s for s in _slides(cache)}
    assert slides["tumor_000"].label == 1 and slides["normal_000"].label == 0
    assert slides["test_000"].split == "test"
    assert slides["test_000"].label == 1 and slides["test_001"].label == 0
    assert slides["tumor_000"].grid[7].tolist() == [1, 1] and slides["tumor_000"].features.shape == (30, 8)


def test_labels_follow_the_overlap_threshold(cache):
    # fixture lesion patches overlap their lesion by 0.5
    assert load_slide(cache, "uni", "tumor_000", label_threshold=0.05).patch_labels.sum() == 4
    assert load_slide(cache, "uni", "tumor_000", label_threshold=0.6).patch_labels.sum() == 0


def test_retention_uses_primary_and_strict_thresholds(cache, tmp_path):
    slides = [load_slide(cache, "uni", s, label_threshold=0.05) for s in slide_ids(cache, "uni")]
    run_method_fold("oracle", 0, slides, CFG, tmp_path, "cpu", {}, log=lambda m: None)
    rows = [r for r in _rows(tmp_path / "oracle" / "fold0" / "K4" / "retention.csv") if r["label"] == "1"]
    assert rows and all(float(r["cov"]) == 1.0 and float(r["lsr"]) == 1.0 for r in rows)  # Oracle keeps all 4
    assert all(r["cov_strict"] == "nan" and float(r["tep_strict"]) == 0.0 for r in rows)  # none >= 0.6


def test_fold_split_is_disjoint_and_excludes_test(cache):
    slides = _slides(cache)
    tr, va = fold_split(slides, 2, 0, 0)
    tr1, va1 = fold_split(slides, 2, 0, 1)
    ids = {s.slide_id for s in tr} | {s.slide_id for s in va}
    assert len(ids) == 12 and not ({s.slide_id for s in tr} & {s.slide_id for s in va})
    assert {s.slide_id for s in va} == {s.slide_id for s in tr1}
    assert not any(s.startswith("test") for s in ids)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_runs_end_to_end(method, cache, tmp_path, monkeypatch):
    # Official AP settings (preference 0) collapse to one prototype on toy data; this test checks the runner,
    # so PMIL gets two fixed prototypes (its clustering is tested in test_pmil.py).
    monkeypatch.setattr(methods, "affinity_prototypes",
                        lambda tops, seed: np.stack([tops[0][0], tops[0][-1]]))
    slides = _slides(cache)
    run_method_fold(method, 0, slides, CFG, tmp_path, "cpu", {"git_sha": "x"}, log=lambda m: None)
    out = tmp_path / method / "fold0" / "K4"
    preds, rets = _rows(out / "predictions.csv"), _rows(out / "retention.csv")
    assert len(preds) == len(rets) == 6 + 4  # one val fold + test
    assert {r["split"] for r in preds} == {"val", "test"}
    assert all(0.0 <= float(r["prob"]) <= 1.0 for r in preds)
    assert all(1 <= int(r["n_kept"]) <= 4 for r in rets)
    done = json.loads((out / "done.json").read_text())
    assert done["method"] == method and done["K"] == 4 and done["n_train"] == 6
    arrays = np.load(out / "retained.npz")
    assert len([k for k in arrays.files if k.startswith("retained/")]) == 10
    if method == "slicechat":
        assert "layer_of_drop/test_000" in arrays.files
        assert arrays["stage_scores/test_000"].shape[1] == 30
    if method in ("abmil", "siimil", "nrk_abmil", "pmil", "lts"):
        assert arrays["scores/test_000"].shape == (30,) and arrays["scores/test_000"].dtype == np.float16
    assert (out / "model.pt").exists()
    if method == "pmil":  # the per-fold max-MIL head before PMIL's per-K fine-tuning (a FROC scorer)
        assert (tmp_path / "pmil" / "fold0" / "maxmil_head.pt").exists()
    if method == "tc_ssa":
        assert "W/test_000" in arrays.files
    if method == "abmil":
        full = _rows(tmp_path / "abmil_full" / "fold0" / "Kfull" / "retention.csv")
        assert all(int(r["n_kept"]) == 30 for r in full)


def test_rerun_skips_finished_budgets(cache, tmp_path):
    slides = _slides(cache)
    run_method_fold("random", 0, slides, CFG, tmp_path, "cpu", {}, log=lambda m: None)
    msgs = []
    run_method_fold("random", 0, slides, CFG, tmp_path, "cpu", {}, log=msgs.append)
    assert msgs == ["random fold 0 K 4: done, skipping"]


@pytest.mark.parametrize("method", ["abmil", "siimil", "lts"])
def test_patch_labels_never_reach_training(method, cache, tmp_path):
    slides = _slides(cache)
    run_method_fold(method, 0, slides, CFG, tmp_path / "a", "cpu", {}, log=lambda m: None)
    scrambled = _slides(cache)
    for s in scrambled:  # wipe every evaluation-only annotation the model could have leaked
        s.patch_labels = np.zeros_like(s.patch_labels)
        s.lesion_ids = np.full_like(s.lesion_ids, -1)
    run_method_fold(method, 0, scrambled, CFG, tmp_path / "b", "cpu", {}, log=lambda m: None)
    a = _rows(tmp_path / "a" / method / "fold0" / "K4" / "predictions.csv")
    b = _rows(tmp_path / "b" / method / "fold0" / "K4" / "predictions.csv")
    assert a == b


@pytest.mark.parametrize("method", METHODS)
def test_external_run_reproduces_the_saved_fold_models(method, cache, tmp_path, monkeypatch):
    # Rebuilding every (method, K) from its saved weights must give the same kept patches and predictions
    # on the same slides as the fitted run did: no retraining happens.
    from dataclasses import replace

    from lesionbench.eval.runner import run_external

    monkeypatch.setattr(methods, "affinity_prototypes",
                        lambda tops, seed: np.stack([tops[0][0], tops[0][-1]]))
    slides = _slides(cache)
    run_method_fold(method, 0, slides, CFG, tmp_path / "src", "cpu", {}, log=lambda m: None)
    external = [replace(s, split="external") for s in slides if s.split == "test"]
    run_external(method, 0, slides, external, CFG, tmp_path / "src", tmp_path / "ext", "cpu", {},
                 log=lambda m: None)
    names = [method] + (["abmil_full"] if method == "abmil" else [])
    for name, K in [(n, "full" if n == "abmil_full" else 4) for n in names]:
        a = tmp_path / "src" / name / "fold0" / f"K{K}"
        b = tmp_path / "ext" / name / "fold0" / f"K{K}"
        src = {r["slide_id"]: r for r in _rows(a / "predictions.csv") if r["split"] == "test"}
        ext = {r["slide_id"]: r for r in _rows(b / "predictions.csv")}
        assert set(ext) == set(src) and {r["split"] for r in ext.values()} == {"external"}
        for sid in src:
            assert float(ext[sid]["prob"]) == pytest.approx(float(src[sid]["prob"]), abs=1e-6)
            np.testing.assert_array_equal(np.load(b / "retained.npz")[f"retained/{sid}"],
                                          np.load(a / "retained.npz")[f"retained/{sid}"])
        assert json.loads((b / "done.json").read_text())["n_external"] == 4


def test_external_analysis_end_to_end(cache, tmp_path, monkeypatch):
    import importlib.util
    from dataclasses import replace

    import yaml

    from lesionbench.config import ROOT
    from lesionbench.eval.runner import run_external

    slides = _slides(cache)
    src_out, ext_out = tmp_path / "src_out", tmp_path / "ext_out"
    run_method_fold("random", 0, slides, CFG, src_out / "runs" / "uni", "cpu", {}, log=lambda m: None)
    external = [replace(s, split="external") for s in slides if s.split == "test"]
    run_external("random", 0, slides, external, CFG, src_out / "runs" / "uni", ext_out / "external" / "uni",
                 "cpu", {}, log=lambda m: None)
    data = tmp_path / "data"
    data.mkdir()
    stage = {"test_000": "micro", "test_001": "negative", "test_002": "itc", "test_003": "negative"}
    (data / "stages.csv").write_text("patient,stage,center\n" + "".join(
        f"{s}.tif,{v},{i % 2}\n" for i, (s, v) in enumerate(stage.items())))
    for name, out, d in (("src", src_out, cache), ("ext", ext_out, cache)):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(
            {"paths": {"data_root": str(data), "cache_root": str(d), "output_root": str(out)}}))
    spec = importlib.util.spec_from_file_location("ea", ROOT / "scripts" / "external_analysis.py")
    ea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ea)
    monkeypatch.setattr("sys.argv", ["ea", "--config", str(tmp_path / "ext.yaml"),
                                     "--source-config", str(tmp_path / "src.yaml")])
    ea.main()
    rows = _rows(ext_out / "analysis" / "uni" / "slide_metrics.csv")
    c17 = {r["group"]: r for r in rows if r["dataset"] == "c17"}
    assert set(c17) == {"all", "0", "1"} and c17["all"]["n"] == "4" and c17["all"]["n_itc"] == "1"
    c16 = [r for r in rows if r["dataset"] == "c16_test"]
    assert len(c16) == 1 and c16[0]["n"] == "4"  # one fold, K = 4, group "all" only
    # same fold model and slides: C16 test and "C17" AUC differ only through the stage labels used
    assert (ext_out / "analysis" / "uni" / "table.md").exists()
    assert _rows(ext_out / "analysis" / "uni" / "lesion_hits.csv")[0]["split"] == "external"


def test_budget_k_fixed_and_percentage():
    from lesionbench.selectors.base import budget_k
    assert budget_k(32, 1000) == 32
    assert budget_k("p30", 1000) == 300 and budget_k("p20", 31) == 6
    assert budget_k("p10", 4) == 1  # never zero
    with pytest.raises(ValueError):
        budget_k("p0", 100)


@pytest.mark.parametrize("method", ["random", "siimil", "nrk_abmil", "slicechat"])
def test_percentage_budget_keeps_that_share_of_each_slide(method, cache, tmp_path):
    slides = _slides(cache)
    run_method_fold(method, 0, slides, CFG, tmp_path, "cpu", {}, budgets=["p30"], log=lambda m: None)
    out = tmp_path / method / "fold0" / "Kp30"
    rets = _rows(out / "retention.csv")
    assert len(rets) == 10 and all(int(r["n_kept"]) == 9 for r in rets)  # round(0.3 * 30)
    assert json.loads((out / "done.json").read_text())["K"] == "p30"


@pytest.mark.parametrize("method", ["pmil", "tc_ssa", "lts"])
def test_fixed_k_methods_reject_percentage_budgets(method, cache):
    from lesionbench.eval.methods import fit
    tr, va = fold_split(_slides(cache), 2, 0, 0)
    with pytest.raises(ValueError, match="fixed K"):
        fit(method, "p30", tr, va, CFG["training"], 0, "cpu", {})
