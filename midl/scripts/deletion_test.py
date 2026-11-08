"""MIDL: lesion-deletion test. Does each compressor's slide call *use* the tumour, or only keep it?

    python midl/scripts/deletion_test.py run --config configs/server.yaml --backbone uni --method lts --fold 0 \
        --out $SCRATCH/lesionbench/outputs_midl/deletion
    python midl/scripts/deletion_test.py summarise --out $SCRATCH/lesionbench/outputs_midl/deletion

Saved CAMELYON16 fold models (runs/<backbone>/<method>/fold<f>/K<K>/model.pt), no training. Each CAMELYON16
test tumour slide is evaluated under five bags; patches are removed from the bag *before* compression, so
nothing of them can reach the slide call through selection, merging or token mixing:

- intact            the full bag (reproduces the run's prediction; checked)
- del_tumour        every patch overlapping any lesion (overlap > 0) removed
- del_tumour_ring   the same plus its 8-neighbour ring (peritumoural tissue)
- del_random        as many random non-tumour, non-ring patches as del_tumour removes (removal control)
- del_dropped       the tumour patches the method did NOT keep on the intact bag removed (kept ones stay).
                    For per-patch top-K selectors the kept set and the call cannot change; a change means
                    the call depends on patches it did not keep (token mixing, slot merging, global scorers).

Calls use the fold's CAMELYON16 threshold (95% specificity on its validation normals). Writes
<out>/<method>_fold<f>.csv; ``summarise`` writes slides.csv (fold-averaged) and summary.md (slide bootstrap).
Reads only; writes only to --out.
"""

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from lesionbench.config import ROOT, load_config
from lesionbench.eval.aggregate import read_csv, write_csv
from lesionbench.eval.data import Slide, load_slide, slide_ids
from lesionbench.eval.external import spec_threshold
from lesionbench.eval.methods import METHODS, fit
from lesionbench.metrics.probe import max_overlap
from lesionbench.splits import make_folds

SPEC = 0.95
CONDITIONS = ("intact", "del_tumour", "del_tumour_ring", "del_random", "del_dropped")
BOOT = 2000


def subset(s: Slide, keep: np.ndarray, tag: str) -> Slide:
    """``s`` restricted to rows where ``keep``; overlaps re-indexed; a new slide_id so no memo/seed is reused."""
    idx = np.flatnonzero(keep)
    remap = np.full(len(keep), -1, np.int64)
    remap[idx] = np.arange(len(idx))
    rows, les, fr = (np.asarray(v) for v in s.overlap)
    m = keep[rows]
    t = torch.from_numpy(idx)
    return replace(s, slide_id=f"{s.slide_id}#{tag}", features=s.features[t], grid=s.grid[t],
                   patch_labels=s.patch_labels[idx], lesion_ids=s.lesion_ids[idx],
                   overlap=(remap[rows[m]], les[m], fr[m]))


def ring(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Patches that are 8-neighbours of a ``mask`` patch and not in ``mask``."""
    occupied = {(int(x), int(y)) for x, y in grid[mask]}
    near = {(x + dx, y + dy) for x, y in occupied for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    return np.array([(int(x), int(y)) in near for x, y in grid]) & ~mask


def bags(s: Slide, kept_intact: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = len(s.features)
    rows, _, fr = (np.asarray(v) for v in s.overlap)
    tumour = np.zeros(n, bool)
    tumour[rows[fr > 0]] = True
    rg = ring(s.grid.numpy(), tumour)
    free = np.flatnonzero(~tumour & ~rg)
    rand = np.ones(n, bool)
    rand[rng.choice(free, size=min(int(tumour.sum()), len(free)), replace=False)] = False
    kept = np.zeros(n, bool)
    kept[kept_intact] = True
    return {"del_tumour": ~tumour, "del_tumour_ring": ~(tumour | rg), "del_random": rand,
            "del_dropped": ~(tumour & ~kept)}


def fold_train_ids(cache: str, backbone: str, cfg: dict, fold: int) -> list[str]:
    """The fold's training slide ids, exactly as runner.fold_split builds them (without loading features)."""
    ex = set(cfg.get("exclude_from_training", []))
    ids = [i for i in slide_ids(cache, backbone) if not i.startswith("test_") and i not in ex]
    tr, _ = make_folds(ids, [int(i.startswith("tumor_")) for i in ids], cfg["training"]["n_folds"],
                       int(cfg["seed"]))[fold]
    return tr


def run(args):
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.method}_fold{args.fold}.csv"
    if dest.exists():
        print(f"{dest} exists, skipping")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache, t = cfg["paths"]["cache_root"], cfg["primary_overlap_threshold"]
    runs = Path(cfg["paths"]["output_root"]) / "runs" / args.backbone
    test = [load_slide(cache, args.backbone, i, label_threshold=t)
            for i in slide_ids(cache, args.backbone) if i.startswith("test_")]
    test = [s for s in test if s.label == 1]
    # Only the normal key banks (SiiMIL, NRK-ABMIL) need training slides: the fold's training normals.
    tr_ids = fold_train_ids(cache, args.backbone, cfg, args.fold)
    train = ([load_slide(cache, args.backbone, i, label_threshold=t) for i in tr_ids if i.startswith("normal_")]
             if args.method in ("siimil", "nrk_abmil") else test[:1])
    tcfg, seed = cfg["training"], int(cfg["seed"]) + args.fold
    names = [(args.method, K) for K in (args.budgets or cfg["budgets"])]
    if args.method == "abmil":
        names.append(("abmil_full", "full"))
    rows, worst, fitcache = [], 0.0, {}
    for name, K in names:
        d = runs / name / f"fold{args.fold}" / f"K{K}"
        preds = read_csv(d / "predictions.csv")
        thr = spec_threshold([float(r["prob"]) for r in preds if r["split"] == "val" and r["label"] == "0"], SPEC)
        stored = {r["slide_id"]: float(r["prob"]) for r in preds if r["split"] == "test"}
        state = torch.load(d / "model.pt", map_location=device)
        fitted = fit(name, 0 if K == "full" else K, train, [], tcfg, seed, device, fitcache, state=state)
        for s in test:
            r = fitted.select(s)
            kept = r.retained_idx.numpy()
            p0 = fitted.prob(s, r)
            worst = max(worst, abs(p0 - stored[s.slide_id]))
            rng = np.random.default_rng([args.fold, int(s.slide_id.split("_")[1])])
            evid = max_overlap(len(s.features), s.overlap[0], s.overlap[2]) >= t
            res = {"intact": (p0, len(s.features), int(evid[kept].sum()))}
            for c, keep in bags(s, kept, rng).items():
                ss = subset(s, keep, c)
                rc = fitted.select(ss)
                ev = max_overlap(len(ss.features), ss.overlap[0], ss.overlap[2]) >= t
                res[c] = (fitted.prob(ss, rc), int((~keep).sum()), int(ev[rc.retained_idx.numpy()].sum()))
            for c, (p, n_rm, n_ev) in res.items():
                rows.append({"method": name, "K": K, "fold": args.fold, "slide_id": s.slide_id,
                             "stratum": s.stratum, "condition": c, "n_patches": len(s.features),
                             "n_removed": 0 if c == "intact" else n_rm, "kept_evidence": n_ev,
                             "prob": p, "threshold": thr, "call": int(p > thr)})
        print(f"{name} fold {args.fold} K {K}: {len(test)} slides; max |intact - stored| = {worst:.2e}", flush=True)
    if worst > 1e-3:
        raise SystemExit(f"intact predictions do not reproduce the stored run (max diff {worst:.3g}); not written")
    write_csv(dest, rows)
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dest.with_suffix(".json").write_text(json.dumps({"git_sha": sha.strip(), "backbone": args.backbone,
        "method": args.method, "fold": args.fold, "device": device, "max_intact_diff": worst}, indent=1))
    print(dest)


def _boot(x: np.ndarray, rng) -> tuple[float, float]:
    b = x[rng.integers(0, len(x), (BOOT, len(x)))].mean(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def summarise(args):
    out = Path(args.out)
    R = [r for f in sorted(out.glob("*_fold*.csv")) for r in read_csv(f)]
    key = lambda r: (r["method"], r["K"], r["slide_id"], r["condition"])
    agg: dict = {}
    for r in R:
        agg.setdefault(key(r), []).append(r)
    S = [{"method": m, "K": K, "slide_id": s, "condition": c, "stratum": rs[0]["stratum"], "n_folds": len(rs),
          "prob": np.mean([float(r["prob"]) for r in rs]), "call": np.mean([int(r["call"]) for r in rs]),
          "n_removed": int(rs[0]["n_removed"]), "kept_evidence": np.mean([int(r["kept_evidence"]) for r in rs])}
         for (m, K, s, c), rs in agg.items()]
    write_csv(out / "slides.csv", S)
    by = {(r["method"], r["K"], r["slide_id"], r["condition"]): r for r in S}
    rng = np.random.default_rng(0)
    order = METHODS + ["abmil_full"]
    lines = ["# Lesion-deletion test (CAMELYON16 test tumour slides)", "",
             "Patches removed from the bag before compression. Call rate = share of slide x fold calls positive "
             "at the fold's CAMELYON16 threshold (95% specificity on validation normals), averaged over the 5 "
             "fold models per slide, then over slides; 95% CI = slide bootstrap (B = 2000). "
             "Delta P = mean change in P(tumour) vs intact.", ""]
    Ks = sorted({r["K"] for r in S}, key=lambda k: (k == "full", int(k) if k != "full" else 0))
    for K in Ks:
        for st in ("micro", "macro"):
            lines += [f"## K = {K}, {st} slides", "",
                      "| method | n | intact call | del_tumour call | del_tumour_ring call | del_random call | "
                      "del_dropped call | Delta P del_tumour | Delta P del_dropped |", "|---|---|---|---|---|---|---|---|---|"]
            for m in order:
                sl = sorted({r["slide_id"] for r in S if r["method"] == m and r["K"] == K and r["stratum"] == st})
                if not sl:
                    continue
                cell = []
                for c in CONDITIONS:
                    x = np.array([by[(m, K, s, c)]["call"] for s in sl])
                    lo, hi = _boot(x, rng)
                    cell.append(f"{x.mean():.2f} [{lo:.2f}, {hi:.2f}]")
                dp = [np.mean([by[(m, K, s, c)]["prob"] - by[(m, K, s, "intact")]["prob"] for s in sl])
                      for c in ("del_tumour", "del_dropped")]
                lines.append(f"| {m} | {len(sl)} | " + " | ".join(cell) + f" | {dp[0]:+.3f} | {dp[1]:+.3f} |")
            lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print(out / "summary.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--config", required=True)
    a.add_argument("--backbone", default="uni")
    a.add_argument("--method", required=True, choices=METHODS)
    a.add_argument("--fold", type=int, required=True)
    a.add_argument("--budgets", type=int, nargs="*")
    a.add_argument("--out", required=True)
    b = sub.add_parser("summarise")
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else summarise(args)


if __name__ == "__main__":
    main()
