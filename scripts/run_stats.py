"""Bootstrap CIs and tests against Random for every method x K on the test slides.

    python scripts/run_stats.py --config configs/server.yaml --backbone uni

Run scripts/aggregate_runs.py first (reads its lesion_hits.csv). Reads OUTPUT_ROOT/runs/<backbone>/ (only
(method, K) with all folds finished) and writes OUTPUT_ROOT/analysis/<backbone>/stats.csv:

- auc / all: fold-mean slide AUC; vs Random by paired bootstrap of the difference.
- cov, lsr, tep / macro, micro, itc: slide means within the slide stratum (per-slide values averaged over
  folds); vs Random by paired Wilcoxon on slides.
- lsr_pooled / macro, micro, itc: lesion-pooled LSR by lesion size class (exhaustive slides); vs Random by
  paired bootstrap of the difference.

CIs: B replicates over slides (seeded, stratified by stratum), each scoring every fold model on the same
resample. Holm within each (metric, stratum, K) over the compared methods. ITC is descriptive only (no p).
"""

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from lesionbench.config import load_config
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

REFERENCE = "random"
COMPARED = ["abmil", "siimil", "nrk_abmil", "pmil", "tc_ssa", "lts", "slicechat"]
DESCRIBED = ["oracle", "abmil_full"]  # ceiling and no-budget reference: estimates and CIs only
STRATA = ["macro", "micro", "itc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="uni")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_folds, B, seed = cfg["training"]["n_folds"], int(cfg["stats"]["bootstrap_iters"]), int(cfg["seed"])
    out = Path(cfg["paths"]["output_root"]) / "analysis" / args.backbone
    folds = defaultdict(dict)
    for method, K, fold, d in run_dirs(Path(cfg["paths"]["output_root"]) / "runs" / args.backbone):
        folds[(method, K)][fold] = d
    runs = {}
    for key, fs in folds.items():
        if key[0] in [REFERENCE] + COMPARED + DESCRIBED:
            if len(fs) == n_folds:
                runs[key] = [fs[f] for f in sorted(fs)]
            else:
                print(f"skipping {key}: {len(fs)}/{n_folds} folds")

    # per run, test split only: probs (F, n) and per-slide retention values averaged over folds
    test_ids = sorted(r["slide_id"] for r in read_csv(runs[(REFERENCE, "32")][0] / "predictions.csv")
                      if r["split"] == "test")
    pos = {s: i for i, s in enumerate(test_ids)}
    y, stratum = np.zeros(len(test_ids), dtype=int), [""] * len(test_ids)
    probs, ret = {}, {}
    for key, dirs in runs.items():
        p = np.full((len(dirs), len(test_ids)), np.nan)
        v = {c: np.full((len(dirs), len(test_ids)), np.nan) for c in ("cov", "lsr", "tep")}
        for f, d in enumerate(dirs):
            for r in read_csv(d / "predictions.csv"):
                if r["split"] == "test":
                    p[f, pos[r["slide_id"]]], y[pos[r["slide_id"]]] = float(r["prob"]), int(r["label"])
            for r in read_csv(d / "retention.csv"):
                if r["split"] == "test":
                    stratum[pos[r["slide_id"]]] = r["stratum"]
                    for c in v:
                        v[c][f, pos[r["slide_id"]]] = float(r[c])
        probs[key] = p
        with warnings.catch_warnings():  # negative slides: Cov / LSR undefined in every fold
            warnings.simplefilter("ignore", RuntimeWarning)
            ret[key] = {c: np.nanmean(a, axis=0) for c, a in v.items()}

    # lesion hits: (F, S) hits and (S,) lesion counts per size class, over test slides with such lesions
    hits = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # (method, K, size) -> fold -> slide
    totals = defaultdict(lambda: defaultdict(int))  # size -> slide -> lesions
    for h in read_csv(out / "lesion_hits.csv"):
        if h["split"] == "test" and h["exhaustive"] == "1" and (h["method"], h["K"]) in runs:
            hits[(h["method"], h["K"], h["size_class"])][h["fold"]][h["slide_id"]] += int(h["hit"])
            if h["method"] == REFERENCE and h["K"] == "32" and h["fold"] == "0":
                totals[h["size_class"]][h["slide_id"]] += 1

    rows = []

    def add(metric, strat, K, method, n, est, boot, ref_boot=None, p=float("nan"), test=""):
        lo, hi = percentile_ci(boot)
        row = {"metric": metric, "stratum": strat, "K": K, "method": method, "n_slides": n, "estimate": est,
               "ci_lo": lo, "ci_hi": hi, "delta_vs_random": float("nan"), "delta_ci_lo": float("nan"),
               "delta_ci_hi": float("nan"), "test": test, "p": p, "p_holm": float("nan")}
        if ref_boot is not None:
            row["delta_vs_random"] = est - rows_ref[(metric, strat, K)]
            row["delta_ci_lo"], row["delta_ci_hi"] = percentile_ci(boot - ref_boot)
        rows.append(row)

    rows_ref = {}
    budgets = sorted({K for _, K in runs if K.isdigit()}, key=int) + ["full"]
    W_all = resample_weights(len(test_ids), B, seed)
    one = np.ones((1, len(test_ids)))
    for K in budgets:
        methods = [m for m in [REFERENCE] + COMPARED + DESCRIBED if (m, K) in runs]
        # AUC over all test slides
        boots = {m: boot_auc(y, probs[(m, K)], W_all) for m in methods}
        for m in methods:
            est = float(boot_auc(y, probs[(m, K)], one)[0])
            if m == REFERENCE:
                rows_ref[("auc", "all", K)] = est
            cmp = m in COMPARED and REFERENCE in boots
            add("auc", "all", K, m, len(test_ids), est, boots[m], boots[REFERENCE] if cmp else None,
                boot_p(boots[m] - boots[REFERENCE]) if cmp else float("nan"),
                "paired bootstrap" if cmp else "")
        # per-slide retention within slide strata
        for s in STRATA:
            cols = np.array([i for i, st in enumerate(stratum) if st == s])
            W = resample_weights(len(cols), B, seed)
            for c in ("cov", "lsr", "tep"):
                vals = {m: ret[(m, K)][c][cols] for m in methods}
                boots = {m: slide_mean(vals[m], W) for m in methods}
                for m in methods:
                    est = float(slide_mean(vals[m], np.ones((1, len(cols))))[0])
                    if m == REFERENCE:
                        rows_ref[(c, s, K)] = est
                    cmp = m in COMPARED and REFERENCE in vals
                    tested = cmp and s != "itc"
                    add(c, s, K, m, len(cols), est, boots[m], boots[REFERENCE] if cmp else None,
                        paired_wilcoxon(vals[m], vals[REFERENCE]) if tested else float("nan"),
                        "paired Wilcoxon" if tested else ("descriptive" if cmp else ""))
            # lesion-pooled LSR by lesion size class
            sl = sorted(totals[s])
            if not sl:
                continue
            tot = np.array([totals[s][x] for x in sl])
            W = resample_weights(len(sl), B, seed)
            mats = {m: np.array([[hits[(m, K, s)][str(f)][x] for x in sl] for f in range(n_folds)])
                    for m in methods if (m, K) in runs}
            boots = {m: pooled_rate(h, tot, W) for m, h in mats.items()}
            for m, h in mats.items():
                est = float(pooled_rate(h, tot, np.ones((1, len(sl))))[0])
                if m == REFERENCE:
                    rows_ref[("lsr_pooled", s, K)] = est
                cmp = m in COMPARED and REFERENCE in mats
                tested = cmp and s != "itc"
                add("lsr_pooled", s, K, m, len(sl), est, boots[m], boots[REFERENCE] if cmp else None,
                    boot_p(boots[m] - boots[REFERENCE]) if tested else float("nan"),
                    "paired bootstrap" if tested else ("descriptive" if cmp else ""))

    family = defaultdict(list)
    for i, r in enumerate(rows):
        if r["method"] in COMPARED:
            family[(r["metric"], r["stratum"], r["K"])].append(i)
    for idx in family.values():
        for i, q in zip(idx, holm([rows[i]["p"] for i in idx]), strict=True):
            rows[i]["p_holm"] = q
    write_csv(out / "stats.csv", rows)
    print(out / "stats.csv", f"B={B} seed={seed}")


if __name__ == "__main__":
    main()
