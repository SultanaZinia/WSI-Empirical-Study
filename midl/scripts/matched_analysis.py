"""MIDL: compression effect at matched specificity on CAMELYON17 (separates compression from threshold shift).

    python midl/scripts/matched_analysis.py --config configs/c17.local.yaml --source-config configs/server.yaml \
        --backbone uni --out $SCRATCH/lesionbench/outputs_midl/matched

Operating-point schemes (thresholds set per fold model and per hospital, on that hospital's negative slides):
- matched_full: each compressed model gets the specificity the full-bag model (abmil_full, same fold) has at
  its CAMELYON16 threshold in that hospital; the full bag keeps its CAMELYON16 threshold.
- spec95 / spec90: every model, full bag included, recalibrated to 95% / 90% specificity per hospital
  ("if recalibrated per hospital": uses CAMELYON17 negative labels).
Plus a threshold-free metric: micro-slide vs negative-slide AUC (per hospital, averaged over the group's hospitals
weighted by micro slides) and its paired drop vs the full bag.

Metrics and bootstrap as patient_analysis.py (patients resampled within hospital, same resamples for every method).
Thresholds are held fixed in the bootstrap (CIs are conditional on the operating point).
Writes summary.csv, auc.csv, table.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_analysis import FULL, SKIP, SPEC, boot_weights, count_vectors, estimate, load_meta  # noqa: E402

from lesionbench.config import load_config  # noqa: E402
from lesionbench.eval.aggregate import read_csv, run_dirs, write_csv  # noqa: E402
from lesionbench.eval.external import spec_threshold  # noqa: E402

AUC_ITERS = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-config", required=True)
    ap.add_argument("--backbone", default="uni")
    ap.add_argument("--centres", default=str(Path(__file__).resolve().parents[1] / "configs" / "c17_centres.yaml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg, src = load_config(args.config), load_config(args.source_config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    iters, seed = int(src["stats"]["bootstrap_iters"]), int(src["seed"])

    centres = yaml.safe_load(open(args.centres))["centres"]
    by = lambda key, val: [c for c, v in centres.items() if v[key] == val]
    groups = {"all": list(centres), "seen": by("seen_in_c16", True), "unseen": by("seen_in_c16", False),
              "new_hospital_same_scanner": by("group", "new_hospital_same_scanner"),
              "new_hospital_new_scanner": by("group", "new_hospital_new_scanner"),
              **{f"centre{c}_{v['hospital']}": [c] for c, v in centres.items()}}

    slides, patients = load_meta(cfg["paths"]["data_root"])
    order = sorted(patients)
    pid = {p: i for i, p in enumerate(order)}
    pn = [patients[p]["pn"] for p in order]
    W = boot_weights(patients, order, groups, iters, seed)
    slide_ids = sorted(slides)
    centre = np.array([slides[s]["centre"] for s in slide_ids])
    stage = np.array([slides[s]["stage"] for s in slide_ids])
    neg = stage == "negative"

    src_runs = {(m, K, f): d for m, K, f, d in run_dirs(Path(src["paths"]["output_root"]) / "runs" / args.backbone)}
    P, T = {}, {}  # (m, K) -> F x S probs, F c16 thresholds
    for m, K, f, d in run_dirs(Path(cfg["paths"]["output_root"]) / "external" / args.backbone):
        if m in SKIP or (m, K, f) not in src_runs:
            continue
        normals = [float(r["prob"]) for r in read_csv(src_runs[(m, K, f)] / "predictions.csv")
                   if r["split"] == "val" and r["label"] == "0"]
        prob = {r["slide_id"]: float(r["prob"]) for r in read_csv(d / "predictions.csv")}
        P.setdefault((m, K), {})[f] = np.array([prob[s] for s in slide_ids])
        T.setdefault((m, K), {})[f] = spec_threshold(normals, SPEC)
    P = {k: np.stack([v[f] for f in sorted(v)]) for k, v in P.items()}
    T = {k: np.array([v[f] for f in sorted(v)]) for k, v in T.items()}
    assert all(p.shape[0] == 5 for p in P.values())

    def calls_at_spec(p, spec_of):  # spec_of(f, c) -> target specificity; per-fold, per-hospital threshold
        called = np.zeros_like(p, dtype=bool)
        for f in range(p.shape[0]):
            for c in centres:
                k = centre == c
                t = spec_threshold(p[f, k & neg], spec_of(f, c))
                called[f, k] = p[f, k] > t
        return called

    pf, tf = P[(FULL, "full")], T[(FULL, "full")]
    full_spec = {(f, c): float(np.mean(pf[f, (centre == c) & neg] <= tf[f])) for f in range(5) for c in centres}
    schemes = {
        "matched_full": lambda m, K: (pf > tf[:, None]) if m == FULL else calls_at_spec(P[(m, K)], lambda f, c: full_spec[(f, c)]),
        "spec95": lambda m, K: calls_at_spec(P[(m, K)], lambda f, c: 0.95),
        "spec90": lambda m, K: calls_at_spec(P[(m, K)], lambda f, c: 0.90),
    }

    rows = []
    for sch, fn in schemes.items():
        counts = {k: count_vectors(fn(*k), slide_ids, slides, pid, pn) for k in sorted(P)}
        full = counts[(FULL, "full")]
        for (m, K), mets in counts.items():
            if m != FULL:
                for name in ("sens_micro", "pt_pn1mi_micro", "pt_pn1mi_any"):
                    mets[f"drop_{name}"] = (full[name][0] - mets[name][0], full[name][1])
            for g, (w, mask) in W.items():
                for name, (num, den) in mets.items():
                    e = estimate(num, den, w, mask)
                    if e is not None:
                        rows.append({"scheme": sch, "method": m, "K": K, "group": g, "metric": name, **e})
    write_csv(out / "summary.csv", rows)

    # threshold-free: micro vs negative AUC, per hospital, combined over the group's hospitals (weights: micro slides)
    rng = np.random.default_rng(seed)
    pat_of = np.array([pid[slides[s]["patient"]] for s in slide_ids])
    sel = (stage == "micro") | neg
    y = (stage == "micro").astype(int)
    pidx_by_c = {c: np.flatnonzero(np.array([patients[p]["centre"] for p in order]) == c) for c in centres}
    boot_pw = []  # per-iteration patient weights (all centres resampled within centre)
    for _ in range(AUC_ITERS):
        w = np.zeros(len(order))
        for c, idx in pidx_by_c.items():
            np.add.at(w, rng.choice(idx, len(idx)), 1.0)
        boot_pw.append(w)

    def hosp_auc(p, c, pw=None):
        k = sel & (centre == c)
        sw = None if pw is None else pw[pat_of[k]]
        if sw is not None and (sw[y[k] == 1].sum() == 0 or sw[y[k] == 0].sum() == 0):
            return np.nan
        return roc_auc_score(y[k], p[k], sample_weight=sw)

    n_micro = {c: int(((stage == "micro") & (centre == c)).sum()) for c in centres}
    keys = [k for k in sorted(P) if k[1] in ("32", "full")]
    # per key, fold, centre: point AUC and AUC_ITERS bootstrap values (computed once, shared by all groups)
    pt_auc, bt_auc = {}, {}
    for key in keys:
        for f in range(5):
            for c in centres:
                if n_micro[c] == 0:
                    continue
                pt_auc[(key, f, c)] = hosp_auc(P[key][f], c)
                bt_auc[(key, f, c)] = np.array([hosp_auc(P[key][f], c, pw) for pw in boot_pw])

    def group_auc(key, cs, boot):
        cs = [c for c in cs if n_micro[c] > 0]
        wts = np.array([n_micro[c] for c in cs], float)
        src_ = bt_auc if boot else pt_auc
        per_fold = [np.ma.average(np.ma.masked_invalid(np.stack([src_[(key, f, c)] for c in cs]) if boot
                                  else np.array([src_[(key, f, c)] for c in cs])), axis=0, weights=wts)
                    for f in range(5)]
        return np.ma.mean(np.ma.stack(per_fold), axis=0).filled(np.nan) if boot else float(np.mean(per_fold))

    auc_rows = []
    fk = (FULL, "full")
    for g, cs in groups.items():
        full_b, full_pt = group_auc(fk, cs, True), group_auc(fk, cs, False)
        for key in keys:
            b, pt = group_auc(key, cs, True), group_auc(key, cs, False)
            d = full_b - b
            auc_rows.append({"method": key[0], "K": key[1], "group": g, "auc_micro": pt,
                             "lo": np.nanpercentile(b, 2.5), "hi": np.nanpercentile(b, 97.5),
                             "drop_vs_full": full_pt - pt, "drop_lo": np.nanpercentile(d, 2.5),
                             "drop_hi": np.nanpercentile(d, 97.5)})
    write_csv(out / "auc.csv", auc_rows)

    idx = {(r["scheme"], r["method"], r["K"], r["group"], r["metric"]): r for r in rows}
    aidx = {(r["method"], r["K"], r["group"]): r for r in auc_rows}
    fmt = lambda r, a="est", lo="lo", hi="hi": f"{r[a]:.2f} [{r[lo]:.2f}, {r[hi]:.2f}]"
    cell = lambda s, m, K, g, c: fmt(idx[(s, m, K, g, c)]) if (s, m, K, g, c) in idx else "–"
    lines = ["# CAMELYON17: compression effect at matched specificity, K = 32 vs full bag", "",
             "Mean over 5 fold models [95% bootstrap CI over patients, resampled within hospital; thresholds held "
             "fixed]. Thresholds per fold model and per hospital. matched_full: compressed model at the full bag's "
             "specificity (full bag at its CAMELYON16 threshold). spec95: all models recalibrated to 95% specificity "
             "per hospital (uses CAMELYON17 negatives). AUC: micro vs negative slides, per hospital, combined "
             f"weighted by micro slides ({AUC_ITERS} resamples).", ""]
    for g in groups:
        lines += [f"## {g} (centres {groups[g]})", "",
                  "| method | AUC micro | AUC drop | matched_full: sens micro | drop | spec95: sens micro | drop | "
                  "spec95: pN1mi (b) | drop | spec95: FA pN0 |", "|---" * 10 + "|"]
        for m, K in sorted(keys, key=lambda x: (x[0] != FULL, x[0])):
            a = aidx[(m, K, g)]
            lines.append(
                f"| {m} K{K} | {fmt(a, 'auc_micro')} | "
                f"{'–' if m == FULL else fmt(a, 'drop_vs_full', 'drop_lo', 'drop_hi')} | "
                f"{cell('matched_full', m, K, g, 'sens_micro')} | {cell('matched_full', m, K, g, 'drop_sens_micro')} | "
                f"{cell('spec95', m, K, g, 'sens_micro')} | {cell('spec95', m, K, g, 'drop_sens_micro')} | "
                f"{cell('spec95', m, K, g, 'pt_pn1mi_micro')} | {cell('spec95', m, K, g, 'drop_pt_pn1mi_micro')} | "
                f"{cell('spec95', m, K, g, 'pt_fa_pn0')} |")
        lines.append("")
    (out / "table.md").write_text("\n".join(lines))
    print(out / "table.md")


if __name__ == "__main__":
    main()
