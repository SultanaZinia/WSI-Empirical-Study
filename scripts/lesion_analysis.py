"""Lesion-level analyses on the test slides at the primary hit threshold (docs/metrics.md Q1, Q3).

    python scripts/lesion_analysis.py --config configs/server.yaml --backbone uni

Run scripts/aggregate_runs.py, scripts/lesion_controls.py and scripts/encoder_audit.py first. Reads their
lesion tables, SLICEChat-style's layer_of_drop (runs/<backbone>/slicechat/) and the patch overlaps
(CACHE_ROOT/patches). Writes OUTPUT_ROOT/analysis/<backbone>/lesion/:

- lesion_outcomes.csv: per method x K x fold x lesion: hit, probe AUC, stage category (cut 0.90).
- d50.csv: diameter at 50% survival (isotonic fit on log d over all size classes, method points pooled over
  folds), per arm x K, with a slide-bootstrap CI.
- micro_gap.csv: micro LSR, Random (exact) and Oracle-LSR-micro, gap to the ceiling and NRS with CIs;
  retention AUC over K.
- attribution.csv: micro lesions per method x K x probe cut (0.85 / 0.90 / 0.95): shares retained / selector
  / encoder / unhittable / unknown
  (lost, no probe AUC), and selector - encoder share among lost lesions (H2) with a CI.
- layer_of_death.csv: SLICEChat-style, fraction of hittable lesions still holding a patch after each pruner.
- summary.md: the micro results at K = 32.

CIs: B slide resamples (seeded); lesions of a slide stay together; fold models share each resample. Only
(method, K) with every fold finished are used.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from lesionbench.config import load_config, write_run_json
from lesionbench.eval.aggregate import read_csv, run_dirs, write_csv
from lesionbench.eval.lesion_analysis import (
    attribute,
    boot_share_diff,
    d50,
    nrs,
    retention_auc,
    survival_by_stage,
)
from lesionbench.stats.inference import percentile_ci, pooled_rate, resample_weights

CONTROLS = ["random", "random_hilbert", "oracle_cov", "oracle_lsr_micro"]
CUTS = [0.85, 0.90, 0.95]
N_PRUNE = 5  # SLICEChat-style pruners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="uni")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t = float(cfg["primary_overlap_threshold"])
    B, seed, n_folds = int(cfg["stats"]["bootstrap_iters"]), int(cfg["seed"]), cfg["training"]["n_folds"]
    root = Path(cfg["paths"]["output_root"])
    out = root / "analysis" / args.backbone / "lesion"
    out.mkdir(parents=True, exist_ok=True)

    n_done = defaultdict(int)
    for m, K, _, _ in run_dirs(root / "runs" / args.backbone):
        n_done[(m, K)] += 1
    complete = {mk for mk, n in n_done.items() if n == n_folds}
    for mk in sorted(set(n_done) - complete):
        print(f"skipping {mk}: {n_done[mk]}/{n_folds} folds")
    hits = [h for h in read_csv(root / "analysis" / args.backbone / "lesion_hits.csv")
            if h["split"] == "test" and h["exhaustive"] == "1" and (h["method"], h["K"]) in complete]
    ctl = [r for r in read_csv(root / "lesion_controls" / "lesions.csv")
           if r["split"] == "test" and r["exhaustive"] == "1" and abs(float(r["threshold"]) - t) < 1e-9
           and r["arm"] in CONTROLS]
    probe = {(r["slide_id"], r["lesion_id"], r["fold"]): float(r["probe_auc"] or "nan")
             for r in read_csv(root / "encoder_audit" / args.backbone / "lesions.csv")
             if r["split"] == "test" and abs(float(r["threshold"]) - t) < 1e-9}

    slides = sorted({h["slide_id"] for h in hits})
    sidx = {s: i for i, s in enumerate(slides)}
    S = len(slides)
    W = resample_weights(S, B, seed)
    one = np.ones((1, S))
    budgets = sorted({h["K"] for h in hits if h["K"].isdigit()}, key=int)
    methods = sorted({h["method"] for h in hits if h["K"].isdigit()})

    # ---- per-lesion outcomes
    outcomes = []
    for h in hits:
        a = probe.get((h["slide_id"], h["lesion_id"], h["fold"]), float("nan"))
        cat = attribute(np.array([a]), np.array([int(h["hittable"])]), np.array([int(h["hit"])]), 0.90)[0]
        keep = ("method", "K", "fold", "slide_id", "lesion_id", "diameter_um", "size_class", "hittable",
                "hit")
        outcomes.append({**{k: h[k] for k in keep}, "probe_auc": a, "category": cat})
    write_csv(out / "lesion_outcomes.csv", outcomes)

    # ---- D50 (all size classes)
    pts = defaultdict(list)  # (arm, K) -> (diameter, hit, slide index)
    for h in hits:
        if h["K"].isdigit():
            pts[(h["method"], h["K"])].append((float(h["diameter_um"]), float(h["hit"]), sidx[h["slide_id"]]))
    for r in ctl:
        if r["slide_id"] in sidx:
            pts[("ctl:" + r["arm"], str(r["K"]))].append((float(r["diameter_um"]), float(r["hit"]),
                                                          sidx[r["slide_id"]]))
    d50_rows = []
    for (arm, K), p in sorted(pts.items()):
        d, y, s = (np.array(c) for c in zip(*p, strict=True))
        boot = np.array([d50(d, y, w[s]) for w in W])
        lo, hi = percentile_ci(boot) if np.isfinite(boot).any() else (float("nan"), float("nan"))
        d50_rows.append({"arm": arm, "K": K, "n_points": len(d), "d50_um": d50(d, y), "ci_lo": lo,
                         "ci_hi": hi, "boot_nan_frac": float(np.mean(~np.isfinite(boot)))})
    write_csv(out / "d50.csv", d50_rows)

    # ---- micro gap to the ceiling and NRS
    micro = [h for h in hits if h["size_class"] == "micro" and h["K"].isdigit()]
    totals = np.zeros(S)
    for h in micro:
        if h["method"] == methods[0] and h["K"] == budgets[0] and h["fold"] == "0":
            totals[sidx[h["slide_id"]]] += 1
    mhits = defaultdict(lambda: np.zeros((n_folds, S)))
    for h in micro:
        mhits[(h["method"], h["K"])][int(h["fold"]), sidx[h["slide_id"]]] += int(h["hit"])
    chits = defaultdict(lambda: np.zeros((1, S)))
    for r in ctl:
        if r["size_class"] == "micro" and r["slide_id"] in sidx:
            chits[(r["arm"], str(r["K"]))][0, sidx[r["slide_id"]]] += float(r["hit"])
    gap_rows, lsr_by = [], defaultdict(dict)
    for K in budgets:
        rnd, orc = chits[("random", K)], chits[("oracle_lsr_micro", K)]
        rb, ob = pooled_rate(rnd, totals, W), pooled_rate(orc, totals, W)
        r0, o0 = pooled_rate(rnd, totals, one)[0], pooled_rate(orc, totals, one)[0]
        for m in methods:
            hb = pooled_rate(mhits[(m, K)], totals, W)
            h0 = pooled_rate(mhits[(m, K)], totals, one)[0]
            lsr_by[m][K] = h0
            nb = np.array([nrs(a, b, c) for a, b, c in zip(hb, rb, ob, strict=True)])
            gap_rows.append({"method": m, "K": K, "n_lesions": int(totals.sum()), "lsr": h0,
                             "lsr_ci": percentile_ci(hb), "random_exact": r0, "oracle_lsr_micro": o0,
                             "gap": o0 - h0, "gap_ci": percentile_ci(ob - hb), "nrs": nrs(h0, r0, o0),
                             "nrs_ci": percentile_ci(nb)})
    for row in gap_rows:
        curve = [lsr_by[row["method"]][k] for k in budgets]
        row["retention_auc"] = retention_auc([int(k) for k in budgets], curve)
        for c in ("lsr_ci", "gap_ci", "nrs_ci"):
            row[c + "_lo"], row[c + "_hi"] = row.pop(c)
    write_csv(out / "micro_gap.csv", gap_rows)

    # ---- encoder vs selector attribution (micro lesions)
    att_rows = []
    by_run = defaultdict(list)
    for o in outcomes:
        if o["size_class"] == "micro" and o["K"].isdigit():
            by_run[(o["method"], o["K"])].append(o)
    for (m, K), rows in sorted(by_run.items()):
        auc = np.array([r["probe_auc"] for r in rows])
        hittable = np.array([int(r["hittable"]) for r in rows])
        hit = np.array([int(r["hit"]) for r in rows])
        fold = np.array([int(r["fold"]) for r in rows])
        sl = np.array([sidx[r["slide_id"]] for r in rows])
        for cut in CUTS:
            cat = attribute(auc, hittable, hit, cut)
            sel, enc = np.zeros((n_folds, S)), np.zeros((n_folds, S))
            np.add.at(sel, (fold[cat == "selector"], sl[cat == "selector"]), 1)
            np.add.at(enc, (fold[cat == "encoder"], sl[cat == "encoder"]), 1)
            diff = boot_share_diff(sel, enc, W)
            lo, hi = percentile_ci(diff)
            att_rows.append({"method": m, "K": K, "cut": cut, "n": len(rows) // n_folds,
                             **{f"share_{c}": float(np.mean(cat == c))
                                for c in ("retained", "selector", "encoder", "unhittable", "unknown")},
                             "selector_minus_encoder_among_lost": float(boot_share_diff(sel, enc, one)[0]),
                             "ci_lo": lo, "ci_hi": hi})
    write_csv(out / "attribution.csv", att_rows)

    # ---- Layer-of-Death (SLICEChat-style)
    overlaps = {}
    for s in slides:
        with h5py.File(Path(cfg["paths"]["cache_root"]) / "patches" / f"{s}.h5") as f:
            overlaps[s] = tuple(f["overlap"][k][:] for k in ("rows", "lesion_ids", "fracs"))
    classes = {(h["slide_id"], int(h["lesion_id"])): h["size_class"] for h in hits}
    lod_rows = []
    for K in [k for k in budgets if ("slicechat", k) in complete]:
        lod = defaultdict(list)
        for f in range(n_folds):
            z = np.load(root / "runs" / args.backbone / "slicechat" / f"fold{f}" / f"K{K}" / "retained.npz")
            for s in slides:
                drop = z[f"layer_of_drop/{s}"]
                rows_, les, fr = overlaps[s]
                for lid in np.unique(les):
                    own = rows_[(les == lid) & (fr >= t)]
                    if len(own) == 0:
                        continue  # unhittable
                    d = drop[own]
                    lod[classes[(s, int(lid))]].append(-1 if (d == -1).any() else int(d.max()))
        for c, v in sorted(lod.items()):
            surv = survival_by_stage(np.array(v), N_PRUNE)
            lod_rows.append({"K": K, "size_class": c, "n_lesion_folds": len(v),
                             **{f"after_pruner_{b}": float(x) for b, x in enumerate(surv)}})
    write_csv(out / "layer_of_death.csv", lod_rows)

    write_run_json(out, cfg, backbone=args.backbone, bootstrap_iters=B, seed=seed, threshold=t)
    (out / "summary.md").write_text(summary(gap_rows, d50_rows, att_rows, lod_rows, budgets[0], B, t))
    print(out / "summary.md")


def summary(gap, d50s, att, lod, K, B, t) -> str:
    f = "{:.3f}".format
    d = {(r["arm"], r["K"]): r for r in d50s}
    lines = [f"# Lesion analyses, test set, K = {K} (hit threshold {t}, B = {B})", "",
             "Micro lesions: LSR, gap to Oracle-LSR-micro, NRS (0 = Random exact, 1 = ceiling). D50 over all "
             "lesions (µm). Stage shares at probe cut 0.90; H2 = selector − encoder share among lost "
             "lesions.", "",
             "| method | micro LSR | NRS [95% CI] | D50 µm [95% CI] | retained | selector | encoder | "
             "H2 [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    for g in [r for r in gap if r["K"] == K]:
        m = g["method"]
        a = next(r for r in att if r["method"] == m and r["K"] == K and r["cut"] == 0.90)
        dd = d.get((m, K), {})
        lines.append(f"| {m} | {f(g['lsr'])} | {f(g['nrs'])} [{f(g['nrs_ci_lo'])}, {f(g['nrs_ci_hi'])}] | "
                     f"{dd.get('d50_um', float('nan')):.0f} [{dd.get('ci_lo', float('nan')):.0f}, "
                     f"{dd.get('ci_hi', float('nan')):.0f}] | {f(a['share_retained'])} | "
                     f"{f(a['share_selector'])} | "
                     f"{f(a['share_encoder'])} | {f(a['selector_minus_encoder_among_lost'])} "
                     f"[{f(a['ci_lo'])}, {f(a['ci_hi'])}] |")
    lines += ["", "Controls D50 (µm): " + ", ".join(
        f"{c} {d[('ctl:' + c, K)]['d50_um']:.0f}" for c in CONTROLS if ("ctl:" + c, K) in d), ""]
    lines += ["SLICEChat-style Layer-of-Death (fraction of hittable lesions alive after each pruner):", ""]
    for r in [r for r in lod if r["K"] == K]:
        lines.append(f"- {r['size_class']} (n = {r['n_lesion_folds']} lesion-folds): " +
                     ", ".join(f(r[f"after_pruner_{b}"]) for b in range(N_PRUNE)))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
