"""MIDL: slide- and patient-level results on CAMELYON17 by hospital group, with the paired drop vs the full bag.

    python midl/scripts/patient_analysis.py --config configs/c17.local.yaml --source-config configs/server.yaml \
        --backbone uni --out $SCRATCH/lesionbench/outputs_midl/patient

Reads only existing results (never writes outside --out):
- source runs  SRC_OUTPUT_ROOT/runs/<backbone>/<method>/fold<f>/K<K>/predictions.csv (CAMELYON16 val normals -> threshold)
- external runs OUTPUT_ROOT/external/<backbone>/<method>/fold<f>/K<K>/predictions.csv
- DATA_ROOT/stages.csv (slide stage, patient pN stage, centre); midl/configs/c17_centres.yaml (hospital groups)

Operating point: as scripts/external_analysis.py (95% specificity on each fold's CAMELYON16 validation normals,
called positive if P > threshold). Oracle is skipped: it reads outlines that exist for 50 of 500 slides only.

Metrics (each a ratio of counts, so all share one bootstrap):
- slide: specificity, sensitivity per stage (itc, micro, macro);
- patient (node-positive = pN1mi, pN1, pN2): positive patients detected (any slide called); pN1mi detected by
  rule (a) any slide called, (b) a micro slide called; false alarm in pN0 and in pN0(i+) patients (any slide called);
- paired drops vs the full bag (abmil_full, same fold): micro-slide sensitivity and pN1mi rule (b).
Point estimate = mean over the 5 fold models; SD over folds; 95% CI from a bootstrap over patients, resampled
within each hospital of the group, the same resamples for every method (paired).

Writes: summary.csv (method x K x group x metric), table.md (K = 32 and full bag).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from lesionbench.config import load_config
from lesionbench.eval.aggregate import read_csv, run_dirs, write_csv
from lesionbench.eval.external import spec_threshold

SPEC = 0.95
FULL = "abmil_full"
SKIP = {"oracle"}
POSITIVE_PN = {"pN1mi", "pN1", "pN2"}


def load_meta(data_root):
    slides, patients = {}, {}
    with open(Path(data_root) / "stages.csv", newline="") as f:
        for r in csv.DictReader(f):
            name = r["patient"]
            if name.endswith(".zip"):
                patients[name[:-4]] = {"pn": r["stage"], "centre": int(r["center"])}
            elif name.endswith(".tif"):
                slides[name[:-4]] = {"stage": r["stage"], "patient": name[:-4].rsplit("_node_", 1)[0],
                                     "centre": int(r["center"])}
    return slides, patients


def count_vectors(called, slide_ids, slides, pid, pn):
    """num (F x P) and den (P) per metric; called: F x S boolean."""
    F, P = called.shape[0], len(pid)
    pidx = np.array([pid[slides[s]["patient"]] for s in slide_ids])
    stage = np.array([slides[s]["stage"] for s in slide_ids])

    def per_patient(mask_slides, values):  # F x P sums over slides
        out = np.zeros((F, P))
        for f in range(F):
            np.add.at(out[f], pidx[mask_slides], values[f][mask_slides])
        return out

    ones = np.ones_like(called, dtype=float)
    m = {}
    neg = stage == "negative"
    m["specificity"] = (per_patient(neg, (~called).astype(float)), per_patient(neg, ones)[0])
    for s in ("itc", "micro", "macro"):
        k = stage == s
        m[f"sens_{s}"] = (per_patient(k, called.astype(float)), per_patient(k, ones)[0])
    all_s = np.ones(len(stage), bool)
    any_called = per_patient(all_s, called.astype(float)) > 0  # F x P
    micro_called = per_patient(stage == "micro", called.astype(float)) > 0
    for name, group in (("pt_pos_detected", POSITIVE_PN), ("pt_pn1mi_any", {"pN1mi"}),
                        ("pt_fa_pn0", {"pN0"}), ("pt_fa_pn0i", {"pN0(i+)"})):
        den = np.array([float(p in group) for p in pn])
        m[name] = (any_called * den, den)
    den = np.array([float(p == "pN1mi") for p in pn])
    m["pt_pn1mi_micro"] = (micro_called * den, den)
    return m


def boot_weights(patients, order, groups, iters, seed):
    rng = np.random.default_rng(seed)
    centre = np.array([patients[p]["centre"] for p in order])
    W = {}
    for g, cs in groups.items():
        w = np.zeros((iters, len(order)))
        for c in cs:
            idx = np.flatnonzero(centre == c)
            draws = rng.integers(0, len(idx), size=(iters, len(idx)))
            for j in range(len(idx)):
                np.add.at(w, (np.arange(iters), idx[draws[:, j]]), 1.0)
        W[g] = (w, np.isin(centre, cs).astype(float))
    return W


def estimate(num, den, w, mask):
    """Point (mean over folds), SD over folds, bootstrap CI. num: F x P, den: P."""
    per_fold = (num * mask) .sum(1) / max((den * mask).sum(), 1e-12) if (den * mask).sum() > 0 else None
    if per_fold is None:
        return None
    b = (w @ num.T) / np.maximum(w @ den, 1e-12)[:, None]  # B x F
    b = b[(w @ den) > 0].mean(1)
    return {"est": float(per_fold.mean()), "sd": float(per_fold.std(ddof=1)) if len(per_fold) > 1 else float("nan"),
            "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "n": float((den * mask).sum())}


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

    src_runs = {(m, K, f): d for m, K, f, d in run_dirs(Path(src["paths"]["output_root"]) / "runs" / args.backbone)}
    ext = {}
    for m, K, f, d in run_dirs(Path(cfg["paths"]["output_root"]) / "external" / args.backbone):
        if m in SKIP or (m, K, f) not in src_runs:
            continue
        normals = [float(r["prob"]) for r in read_csv(src_runs[(m, K, f)] / "predictions.csv")
                   if r["split"] == "val" and r["label"] == "0"]
        t = spec_threshold(normals, SPEC)
        prob = {r["slide_id"]: float(r["prob"]) for r in read_csv(d / "predictions.csv")}
        ext.setdefault((m, K), {})[f] = (t, prob)

    slide_ids = sorted(slides)
    counts = {}
    for (m, K), folds in sorted(ext.items()):
        fs = sorted(folds)
        assert len(fs) == 5 and all(set(folds[f][1]) == set(slide_ids) for f in fs), (m, K, fs)
        called = np.array([[folds[f][1][s] > folds[f][0] for s in slide_ids] for f in fs])
        counts[(m, K)] = count_vectors(called, slide_ids, slides, pid, pn)

    full = counts[(FULL, "full")]
    rows = []
    for (m, K), mets in sorted(counts.items()):
        if m != FULL:
            for name in ("sens_micro", "pt_pn1mi_micro"):
                (nf, d), (nm, _) = full[name], mets[name]
                mets[f"drop_{name}"] = (nf - nm, d)
        for g, (w, mask) in W.items():
            for name, (num, den) in mets.items():
                e = estimate(num, den, w, mask)
                if e is not None:
                    rows.append({"method": m, "K": K, "group": g, "metric": name, **e})
    write_csv(out / "summary.csv", rows)

    idx = {(r["method"], r["K"], r["group"], r["metric"]): r for r in rows}
    cell = lambda m, K, g, c: (f"{idx[(m, K, g, c)]['est']:.2f} [{idx[(m, K, g, c)]['lo']:.2f}, "
                               f"{idx[(m, K, g, c)]['hi']:.2f}]") if (m, K, g, c) in idx else "–"
    cols = [("specificity", "Spec"), ("sens_micro", "Sens micro"), ("drop_sens_micro", "Drop micro vs full"),
            ("pt_pos_detected", "Pos. patients"), ("pt_pn1mi_any", "pN1mi (a) any"),
            ("pt_pn1mi_micro", "pN1mi (b) micro"), ("drop_pt_pn1mi_micro", "Drop pN1mi (b)"),
            ("pt_fa_pn0", "FA pN0"), ("pt_fa_pn0i", "FA pN0(i+)")]
    lines = ["# CAMELYON17 slide and patient level, K = 32 vs full bag", "",
             "Mean over 5 fold models [95% bootstrap CI over patients, resampled within hospital]. Threshold: 95% "
             "specificity on each fold's CAMELYON16 validation normals. Oracle excluded (outlines on 50 slides only). "
             "CAMELYON17 patients are artificial 5-slide groupings from one centre.", ""]
    for g in groups:
        n = lambda c: int(idx[(FULL, "full", g, c)]["n"]) if (FULL, "full", g, c) in idx else 0
        lines += [f"## {g} (centres {groups[g]}; micro slides {n('sens_micro')}, pN1mi patients "
                  f"{n('pt_pn1mi_micro')}, pN0 {n('pt_fa_pn0')})", "",
                  "| method | " + " | ".join(h for _, h in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for m, K in sorted(counts, key=lambda x: (x[0] != FULL, x[0])):
            if K in ("32", "full"):
                lines.append(f"| {m} K{K} | " + " | ".join(cell(m, K, g, c) for c, _ in cols) + " |")
        lines.append("")
    (out / "table.md").write_text("\n".join(lines))
    (out / "run.json").write_text(json.dumps({"spec": SPEC, "iters": iters, "seed": seed, "groups": groups,
                                              "config": args.config, "source_config": args.source_config}, indent=1))
    print(out / "table.md")


if __name__ == "__main__":
    main()
