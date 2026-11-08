"""MIDL Part B: rule-based slide category (negative / ITC / micro / macro) from each method's kept patches,
then patient pN stage and quadratic weighted kappa on CAMELYON17 (the CAMELYON17 challenge metric).

    python midl/scripts/nodal_stage.py --c16-config configs/server.yaml --c17-config configs/c17.local.yaml \
        --backbone uni --out $SCRATCH/lesionbench/outputs_midl/nodal

Rule (fixed before looking at CAMELYON17 categories; no parameter is fitted on CAMELYON17):
1. Slide call: the method's own slide probability vs its threshold. Two operating points:
   c16 = 95% specificity on the fold's CAMELYON16 validation normals (deployment);
   spec95 = 95% specificity on the hospital's own negative slides (recalibrated; uses CAMELYON17 labels).
   Called negative -> category "negative".
2. Called positive -> tumour patches = kept patches with shared ABMIL scorer probability >= TAU (the fold's
   full-bag ABMIL patch head, as for FROC). Largest 8-connected cluster on the patch grid; its extent =
   (max span in patches + 1) x patch side (128 um at 0.5 um/px, 256 px). Extent <= 0.2 mm -> ITC,
   <= 2 mm -> micro, else macro (AJCC). No tumour patch kept -> ITC (positive call, no measurable focus).
3. Patient pN (CAMELYON17 rules): pN0 all negative; pN0(i+) only ITC; pN1mi micro but no macro;
   pN1 1-3 positive nodes (micro/macro) with >= 1 macro; pN2 4+ positive nodes with >= 1 macro.
Kappa: quadratic weighted over pN0 < pN0(i+) < pN1mi < pN1 < pN2; mean over 5 fold models; 95% CI from a
bootstrap over patients resampled within hospital. Sanity check without CAMELYON17: slide-category accuracy on
the CAMELYON16 test set (c16 operating point).
Reads only; writes only to --out.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import ndimage
from sklearn.metrics import cohen_kappa_score

from lesionbench.config import load_config
from lesionbench.eval.aggregate import read_csv, run_dirs, write_csv
from lesionbench.eval.data import load_slide, slide_ids
from lesionbench.eval.external import spec_threshold
from lesionbench.selectors.abmil import GatedABMIL

SPEC = 0.95
TAU = 0.5
PATCH_UM = 128.0
CATS = ["negative", "itc", "micro", "macro"]
PN = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]
SKIP = {"oracle"}
ITERS = 2000


def category(kept_idx, p_patch, grid):
    pos = kept_idx[p_patch[kept_idx] >= TAU]
    if len(pos) == 0:
        return "itc", 0.0
    g = grid[pos] - grid[pos].min(0)
    img = np.zeros(g.max(0) + 1, bool)
    img[g[:, 0], g[:, 1]] = True
    lab, n = ndimage.label(img, structure=np.ones((3, 3)))
    best = 0.0
    for c in range(1, n + 1):
        ys, xs = np.nonzero(lab == c)
        best = max(best, (max(ys.max() - ys.min(), xs.max() - xs.min()) + 1) * PATCH_UM)
    mm = best / 1000
    return ("itc" if mm <= 0.2 else "micro" if mm <= 2.0 else "macro"), mm


def pn_stage(cats):
    macro, micro, itc = (sum(c == k for c in cats) for k in ("macro", "micro", "itc"))
    if macro:
        return "pN1" if macro + micro <= 3 else "pN2"
    if micro:
        return "pN1mi"
    return "pN0(i+)" if itc else "pN0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c16-config", required=True)
    ap.add_argument("--c17-config", required=True)
    ap.add_argument("--backbone", default="uni")
    ap.add_argument("--centres", default=str(Path(__file__).resolve().parents[1] / "configs" / "c17_centres.yaml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, tgt = load_config(args.c16_config), load_config(args.c17_config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(src["seed"]))
    nf = src["training"]["n_folds"]

    src_root = Path(src["paths"]["output_root"]) / "runs" / args.backbone
    ext_root = Path(tgt["paths"]["output_root"]) / "external" / args.backbone
    src_runs = {(m, K, f): d for m, K, f, d in run_dirs(src_root) if m not in SKIP}
    ext_runs = {(m, K, f): d for m, K, f, d in run_dirs(ext_root) if m not in SKIP}
    keys = sorted({(m, K) for m, K, _ in ext_runs})

    # thresholds, slide probabilities, kept indices
    thr = {(m, K, f): spec_threshold([float(r["prob"]) for r in read_csv(d / "predictions.csv")
                                      if r["split"] == "val" and r["label"] == "0"], SPEC)
           for (m, K, f), d in src_runs.items()}
    prob17 = {k: {r["slide_id"]: float(r["prob"]) for r in read_csv(d / "predictions.csv")} for k, d in ext_runs.items()}
    prob16 = {k: {r["slide_id"]: float(r["prob"]) for r in read_csv(d / "predictions.csv") if r["split"] == "test"}
              for k, d in src_runs.items()}
    kept17 = {k: np.load(d / "retained.npz") for k, d in ext_runs.items()}
    kept16 = {k: np.load(d / "retained.npz") for k, d in src_runs.items()}

    scorers = []
    for f in range(nf):
        st = torch.load(src_root / "abmil_full" / f"fold{f}" / "Kfull" / "model.pt", map_location="cpu")
        m = GatedABMIL(st["fc.0.weight"].shape[1], dropout=src["training"]["dropout"])
        m.load_state_dict(st)
        m.eval()
        scorers.append(m)

    @torch.no_grad()
    def patch_probs(x):
        return [torch.softmax(m.classifier(m.fc(x)), 1)[:, 1].numpy() for m in scorers]

    # metadata
    import csv
    slides17, patients = {}, {}
    with open(Path(tgt["paths"]["data_root"]) / "stages.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["patient"].endswith(".zip"):
                patients[r["patient"][:-4]] = {"pn": r["stage"], "centre": int(r["center"])}
            elif r["patient"].endswith(".tif"):
                slides17[r["patient"][:-4]] = {"stage": r["stage"], "centre": int(r["center"]),
                                               "patient": r["patient"][:-4].rsplit("_node_", 1)[0]}
    stage16 = {r["slide_id"]: r["stratum"] for r in read_csv(src_root / "abmil_full" / "fold0" / "Kfull" / "retention.csv")
               if r["split"] == "test"}

    # spec95 thresholds per (m, K, f, centre) on CAMELYON17 negatives
    thr95 = {}
    for (m, K, f), p in prob17.items():
        for c in {v["centre"] for v in slides17.values()}:
            neg = [p[s] for s, v in slides17.items() if v["centre"] == c and v["stage"] == "negative"]
            thr95[(m, K, f, c)] = spec_threshold(neg, SPEC)

    # slide categories
    rows = []
    c16 = src["paths"]["cache_root"]
    for sid in [i for i in slide_ids(c16, args.backbone) if i.startswith("test_")]:
        s = load_slide(c16, args.backbone, sid)
        pp, grid = patch_probs(s.features.float()), s.grid.numpy()
        for (m, K) in keys:
            for f in range(nf):
                if (m, K, f) not in src_runs:
                    continue
                kept = kept16[(m, K, f)][f"retained/{sid}"]
                called = prob16[(m, K, f)][sid] > thr[(m, K, f)]
                cat, mm = category(kept, pp[f], grid) if called else ("negative", 0.0)
                rows.append({"dataset": "c16_test", "slide_id": sid, "method": m, "K": K, "fold": f, "op": "c16",
                             "truth": stage16[sid], "pred": cat, "extent_mm": mm, "centre": ""})
        print("c16", sid, flush=True)
    c17 = tgt["paths"]["cache_root"]
    for sid in sorted(slides17):
        s = load_slide(c17, args.backbone, sid)
        pp, grid = patch_probs(s.features.float()), s.grid.numpy()
        c = slides17[sid]["centre"]
        for (m, K) in keys:
            for f in range(nf):
                kept = kept17[(m, K, f)][f"retained/{sid}"]
                cat_pos, mm = category(kept, pp[f], grid)
                for op, t in (("c16", thr[(m, K, f)]), ("spec95", thr95[(m, K, f, c)])):
                    called = prob17[(m, K, f)][sid] > t
                    rows.append({"dataset": "c17", "slide_id": sid, "method": m, "K": K, "fold": f, "op": op,
                                 "truth": slides17[sid]["stage"], "pred": cat_pos if called else "negative",
                                 "extent_mm": mm if called else 0.0, "centre": c})
        print("c17", sid, flush=True)
    write_csv(out / "slide_categories.csv", rows)

    # patient pN and kappa
    centres = yaml.safe_load(open(args.centres))["centres"]
    groups = {"all": list(centres), "seen": [c for c, v in centres.items() if v["seen_in_c16"]],
              "unseen": [c for c, v in centres.items() if not v["seen_in_c16"]]}
    order = sorted(patients)
    pc = np.array([patients[p]["centre"] for p in order])
    truth = np.array([PN.index(patients[p]["pn"]) for p in order])
    boots = {}
    for g, cs in groups.items():
        W = np.zeros((ITERS, len(order)))
        for c in cs:
            idx = np.flatnonzero(pc == c)
            for b in range(ITERS):
                np.add.at(W[b], rng.choice(idx, len(idx)), 1.0)
        boots[g] = W

    pred_cat = {(r["method"], r["K"], r["fold"], r["op"], r["slide_id"]): r["pred"] for r in rows if r["dataset"] == "c17"}
    prow, krow = [], []
    for (m, K) in keys:
        for op in ("c16", "spec95"):
            P = np.zeros((nf, len(order)), int)
            for f in range(nf):
                for i, p in enumerate(order):
                    cats = [pred_cat[(m, K, f, op, f"{p}_node_{n}")] for n in range(5)]
                    P[f, i] = PN.index(pn_stage(cats))
                    prow.append({"method": m, "K": K, "fold": f, "op": op, "patient": p,
                                 "truth": PN[truth[i]], "pred": PN[P[f, i]]})
            for g, cs in groups.items():
                mask = np.isin(pc, cs)
                kap = lambda w: np.mean([cohen_kappa_score(truth, P[f], weights="quadratic", sample_weight=w,
                                                           labels=list(range(5))) for f in range(nf)])
                est = kap(mask.astype(float))
                bs = np.array([kap(w) for w in boots[g][:: max(1, ITERS // 500)]])  # 500 resamples for speed
                exact = np.mean([(P[f][mask] == truth[mask]).mean() for f in range(nf)])
                krow.append({"method": m, "K": K, "op": op, "group": g, "kappa": est,
                             "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5), "exact_stage_acc": exact})
    write_csv(out / "patient_stages.csv", prow)
    write_csv(out / "kappa.csv", krow)

    # summary
    lines = ["# Part B: rule-based slide category -> patient pN -> quadratic weighted kappa (CAMELYON17)", "",
             f"Tumour patches: kept patches with shared ABMIL scorer p >= {TAU}; largest 8-connected cluster extent "
             f"(patch {PATCH_UM:.0f} um) -> AJCC category. Mean over 5 folds [95% CI, 500 patient resamples within "
             "hospital]. c16 = CAMELYON16 threshold; spec95 = recalibrated per hospital (uses CAMELYON17 negatives).", ""]
    kidx = {(r["method"], r["K"], r["op"], r["group"]): r for r in krow}
    for op in ("c16", "spec95"):
        lines += [f"## Operating point: {op}", "", "| method | all | seen | unseen | exact stage (all) |", "|---|---|---|---|---|"]
        for m, K in sorted(keys, key=lambda x: (x[0] != "abmil_full", x[0], x[1])):
            if K not in ("32", "full"):
                continue
            c = lambda g: f"{kidx[(m, K, op, g)]['kappa']:.2f} [{kidx[(m, K, op, g)]['lo']:.2f}, {kidx[(m, K, op, g)]['hi']:.2f}]"
            lines.append(f"| {m} K{K} | {c('all')} | {c('seen')} | {c('unseen')} | {kidx[(m, K, op, 'all')]['exact_stage_acc']:.2f} |")
        lines.append("")
    lines += ["## Sanity check: slide-category accuracy on CAMELYON16 test (c16 threshold, no CAMELYON17 involved)", "",
              "| method | negative | micro | macro | all |", "|---|---|---|---|---|"]
    for m, K in sorted(keys, key=lambda x: (x[0] != "abmil_full", x[0], x[1])):
        if K not in ("32", "full"):
            continue
        rs = [r for r in rows if r["dataset"] == "c16_test" and r["method"] == m and r["K"] == K]
        acc = lambda st: np.mean([r["pred"] == r["truth"] for r in rs if st is None or r["truth"] == st]) if rs else np.nan
        lines.append(f"| {m} K{K} | {acc('negative'):.2f} | {acc('micro'):.2f} | {acc('macro'):.2f} | {acc(None):.2f} |")
    lines += ["", "## CAMELYON17 slide-category confusion, abmil_full and abmil K32 (c16 / spec95, pooled folds)", ""]
    for m, K in (("abmil_full", "full"), ("abmil", "32")):
        for op in ("c16", "spec95"):
            rs = [r for r in rows if r["dataset"] == "c17" and r["method"] == m and r["K"] == K and r["op"] == op]
            lines += [f"### {m} K{K}, {op} (rows truth, columns predicted)", "", "| | " + " | ".join(CATS) + " |",
                      "|---" * 5 + "|"]
            for t in CATS:
                lines.append(f"| {t} | " + " | ".join(str(sum(r["truth"] == t and r["pred"] == p for r in rs)) for p in CATS) + " |")
            lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print(out / "summary.md")


if __name__ == "__main__":
    main()
