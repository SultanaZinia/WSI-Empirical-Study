"""Paired encoder comparison: the same method, K and test slides with two backbones (default CONCH - UNI).

    python scripts/encoder_compare.py --config configs/server.yaml --backbones uni conch

Run scripts/aggregate_runs.py for both backbones first (reads analysis/<backbone>/lesion_hits.csv). Writes
OUTPUT_ROOT/analysis/encoder_compare/<a>_vs_<b>.csv, one row per (metric, stratum, K, method) with each
backbone's estimate, the difference b - a, its 95% CI and p:

- auc / all: fold-mean slide AUC; paired bootstrap of the difference (same slide resample for both).
- cov, lsr, tep / macro, micro: per-slide values (fold-averaged), paired Wilcoxon on slides.
- micro_any / micro: per micro-stratum slide, share of folds keeping >= 1 of its micro lesions; paired Wilcoxon.
- lsr_pooled / macro, micro: lesion-pooled LSR by lesion size class; paired bootstrap of the difference.

Pairing is valid because both backbones use the same patch grid, folds and test slides (checked below). Holm
within each (metric, stratum, K) over the compared methods. The no-compression reference is described only.
ITC: no test slides in that stratum; ITC lesions are descriptive only (no p).
"""

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from lesionbench.config import load_config, write_run_json
from lesionbench.eval.aggregate import read_csv, run_dirs, write_csv
from lesionbench.stats.inference import (
    boot_auc,
    boot_p,
    holm,
    paired_wilcoxon,
    percentile_ci,
    pooled_rate,
    resample_weights,
    slide_mean,
)

# published compressors (as run_stats)
COMPARED = ["siimil", "nrk_abmil", "pmil", "tc_ssa", "lts", "slicechat", "dtfd", "histoselect"]
DESCRIBED = ["abmil_full"]  # no-compression reference: estimates only
STRATA = ["macro", "micro", "itc"]


def load(root: Path, backbone: str, n_folds: int) -> dict:
    """Test-split predictions, per-slide retention, slide strata and lesion hits for one backbone."""
    folds = defaultdict(dict)
    for method, K, fold, d in run_dirs(root / "runs" / backbone):
        if method in COMPARED + DESCRIBED:
            folds[(method, K)][fold] = d
    runs = {k: [fs[f] for f in sorted(fs)] for k, fs in folds.items() if len(fs) == n_folds}
    probs, ret, y, stratum = {}, {}, {}, {}
    for key, dirs in runs.items():
        p, v = defaultdict(dict), {c: defaultdict(dict) for c in ("cov", "lsr", "tep")}
        for f, d in enumerate(dirs):
            for r in read_csv(d / "predictions.csv"):
                if r["split"] == "test":
                    p[r["slide_id"]][f], y[r["slide_id"]] = float(r["prob"]), int(r["label"])
            for r in read_csv(d / "retention.csv"):
                if r["split"] == "test":
                    stratum[r["slide_id"]] = r["stratum"]
                    for c in v:
                        v[c][r["slide_id"]][f] = float(r[c])
        probs[key], ret[key] = p, v
    hits = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # (method, K, size) -> fold -> slide
    totals = defaultdict(lambda: defaultdict(int))  # size -> slide -> lesions (from one run)
    first = None
    for h in read_csv(root / "analysis" / backbone / "lesion_hits.csv"):
        if h["split"] == "test" and h["exhaustive"] == "1" and (h["method"], h["K"]) in runs:
            hits[(h["method"], h["K"], h["size_class"])][h["fold"]][h["slide_id"]] += int(h["hit"])
            first = first or (h["method"], h["K"])
            if (h["method"], h["K"]) == first and h["fold"] == "0":
                totals[h["size_class"]][h["slide_id"]] += 1
    return {"runs": runs, "probs": probs, "ret": ret, "y": y, "stratum": stratum, "hits": hits, "totals": totals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbones", nargs=2, default=["uni", "conch"], help="a b; differences are b - a")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_folds, B, seed = cfg["training"]["n_folds"], int(cfg["stats"]["bootstrap_iters"]), int(cfg["seed"])
    root = Path(cfg["paths"]["output_root"])
    a, b = args.backbones
    A, Bk = load(root, a, n_folds), load(root, b, n_folds)

    # pairing checks: same test slides, labels, strata and lesions under both backbones
    assert A["y"] == Bk["y"] and A["stratum"] == Bk["stratum"], "test slides / labels / strata differ"
    assert {s: dict(t) for s, t in A["totals"].items()} == {s: dict(t) for s, t in Bk["totals"].items()}, \
        "lesion sets differ"
    test_ids = sorted(A["y"])
    y = np.array([A["y"][s] for s in test_ids])
    stratum = [A["stratum"][s] for s in test_ids]
    pairs = sorted(set(A["runs"]) & set(Bk["runs"]), key=lambda k: (k[1] != "full", k[1].zfill(6), k[0]))
    for k in sorted(set(A["runs"]) ^ set(Bk["runs"])):
        print(f"skipping {k}: complete for one backbone only")

    def mat(per_slide, ids):  # {slide: {fold: value}} -> (F, n)
        return np.array([[per_slide.get(s, {}).get(f, np.nan) for s in ids] for f in range(n_folds)])

    def fold_mean(per_slide, ids):
        with warnings.catch_warnings():  # negative slides: Cov / LSR undefined in every fold
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(mat(per_slide, ids), axis=0)

    rows = []

    def add(metric, strat, K, method, n, ea, eb, boot_a, boot_b, p, test):
        lo, hi = percentile_ci(boot_b - boot_a)
        rows.append({"metric": metric, "stratum": strat, "K": K, "method": method, "n": n, a: ea, b: eb,
                     "delta": eb - ea, "delta_ci_lo": lo, "delta_ci_hi": hi, "test": test, "p": p,
                     "p_holm": float("nan")})

    W_all = resample_weights(len(test_ids), B, seed)
    one = np.ones((1, len(test_ids)))
    for m, K in pairs:
        tested = m in COMPARED
        pa, pb = mat(A["probs"][(m, K)], test_ids), mat(Bk["probs"][(m, K)], test_ids)
        ba, bb = boot_auc(y, pa, W_all), boot_auc(y, pb, W_all)
        add("auc", "all", K, m, len(test_ids), float(boot_auc(y, pa, one)[0]), float(boot_auc(y, pb, one)[0]),
            ba, bb, boot_p(bb - ba) if tested else float("nan"), "paired bootstrap" if tested else "")
        if K == "full":
            continue
        for s in STRATA:
            ids = [t for t, st in zip(test_ids, stratum, strict=True) if st == s]
            W, ones = resample_weights(len(ids), B, seed), np.ones((1, len(ids)))
            per_slide = {c: (fold_mean(A["ret"][(m, K)][c], ids), fold_mean(Bk["ret"][(m, K)][c], ids))
                         for c in ("cov", "lsr", "tep")} if ids else {}
            if s == "micro" and ids:  # >= 1 micro lesion kept on micro-stratum slides, share of folds
                anyk = [np.array([[float(X["hits"][(m, K, "micro")][str(f)].get(t, 0) > 0) for t in ids]
                                  for f in range(n_folds)]).mean(axis=0) for X in (A, Bk)]
                per_slide["micro_any"] = tuple(anyk)
            for c, (va, vb) in per_slide.items():
                t_ = tested and s != "itc"
                add(c, s, K, m, len(ids), float(slide_mean(va, ones)[0]), float(slide_mean(vb, ones)[0]),
                    slide_mean(va, W), slide_mean(vb, W), paired_wilcoxon(vb, va) if t_ else float("nan"),
                    "paired Wilcoxon" if t_ else "descriptive")
            sl = sorted(A["totals"][s])
            if not sl:
                continue
            tot, Wl, onel = np.array([A["totals"][s][x] for x in sl]), resample_weights(len(sl), B, seed), \
                np.ones((1, len(sl)))
            ha, hb = (np.array([[X["hits"][(m, K, s)][str(f)][x] for x in sl] for f in range(n_folds)])
                      for X in (A, Bk))
            ba, bb = pooled_rate(ha, tot, Wl), pooled_rate(hb, tot, Wl)
            t_ = tested and s != "itc"
            add("lsr_pooled", s, K, m, len(sl), float(pooled_rate(ha, tot, onel)[0]),
                float(pooled_rate(hb, tot, onel)[0]), ba, bb, boot_p(bb - ba) if t_ else float("nan"),
                "paired bootstrap" if t_ else "descriptive")

    family = defaultdict(list)
    for i, r in enumerate(rows):
        if r["method"] in COMPARED:
            family[(r["metric"], r["stratum"], r["K"])].append(i)
    for idx in family.values():
        for i, q in zip(idx, holm([rows[i]["p"] for i in idx]), strict=True):
            rows[i]["p_holm"] = q
    out = root / "analysis" / "encoder_compare"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / f"{a}_vs_{b}.csv", rows)
    write_run_json(out, cfg, f"{a}_vs_{b}.json", backbones=[a, b], bootstrap_iters=B, seed=seed,
                   compared=COMPARED)
    print(out / f"{a}_vs_{b}.csv", f"B={B} seed={seed}")


if __name__ == "__main__":
    main()
