"""MIDL: do the lesions that Top-K ABMIL drops matter to the full-bag ABMIL decision?

    python midl/scripts/attention_check.py --c16-config configs/server.yaml --c17-config configs/c17.local.yaml \
        --backbone uni --K 32 --out $SCRATCH/lesionbench/outputs_midl/attention

Top-K ABMIL is the full-bag ABMIL model keeping its own K highest-attention patches (ABMILTopK reuses the same
fitted model), so a lesion is dropped exactly when none of its patches is in the full model's top K.
Its classifier is linear, so the full-bag slide logit difference (tumour - normal) is exactly
sum_i a_i d_i (a = softmax attention, d_i = patch logit difference): each lesion's contribution is exact.

Per fold model (CAMELYON16 weights), on slides with outlined lesions (CAMELYON16 test; CAMELYON17 annotated):
- per lesion: kept at K (a patch with overlap >= primary threshold in the top K), attention mass and logit
  contribution of its patches (overlap >= primary threshold), best attention rank;
- per slide: P(full), P(top K only), P(full without the patches of dropped lesions), P(full without the
  patches of kept lesions) (positive control), P(full without any lesion patch); calls at the fold's
  CAMELYON16 threshold (95% specificity on validation normals). Removed = any overlap > 0.
Writes lesions.csv, slides.csv, summary.md. Reads only; writes only to --out.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lesionbench.config import load_config
from lesionbench.eval.aggregate import read_csv, write_csv
from lesionbench.eval.data import load_slide, slide_ids
from lesionbench.eval.external import spec_threshold
from lesionbench.selectors.abmil import GatedABMIL

SPEC = 0.95


def size_class(d_mm, itc_max, micro_max):
    return "itc" if d_mm <= itc_max else "micro" if d_mm <= micro_max else "macro"


@torch.no_grad()
def prob(model, h, keep):
    """P(tumour) of the bag restricted to boolean mask ``keep`` (h = fc(features), precomputed)."""
    scores = model.attn_c(model.attn_a(h[keep]) * model.attn_b(h[keep])).squeeze(-1)
    pooled = torch.softmax(scores, 0) @ h[keep]
    return float(torch.softmax(model.classifier(pooled), 0)[1])


@torch.no_grad()
def analyse(dataset, s, model, K, t, primary, strata, fold):
    lesion_rows, slide_rows = [], []
    for _ in [0]:
        diam = s.lesion_diameter_um
        if len(diam) == 0:
            continue
        x = s.features.float()
        h = model.fc(x)
        scores = model.attn_c(model.attn_a(h) * model.attn_b(h)).squeeze(-1)
        a = torch.softmax(scores, 0).double().numpy()
        z = model.classifier(h)
        d = (z[:, 1] - z[:, 0]).double().numpy()
        n = len(a)
        order = np.argsort(-scores.numpy(), kind="stable")
        rank = np.empty(n, int)
        rank[order] = np.arange(n)
        top = np.zeros(n, bool)
        top[order[:min(K, n)]] = True
        rows, les, fracs = (np.asarray(v) for v in s.overlap)
        total = float((a * d).sum())
        kept_ids, lost_ids = [], []
        for lid, dm in enumerate(diam):
            elig = np.unique(rows[(les == lid) & (fracs >= primary)])
            if len(elig) == 0:
                continue  # not hittable at the primary threshold
            kept = bool(top[elig].any())
            (kept_ids if kept else lost_ids).append(lid)
            lesion_rows.append({
                "dataset": dataset, "fold": fold, "slide_id": s.slide_id, "lesion_id": lid,
                "diameter_um": float(dm), "size_class": size_class(dm / 1000, strata["itc_max"], strata["micro_max"]),
                "n_patches": len(elig), "kept": int(kept), "best_rank": int(rank[elig].min()),
                "attn_mass": float(a[elig].sum()), "contrib": float((a[elig] * d[elig]).sum()),
                "slide_logit_diff": total})
        any_patch = lambda ids: np.isin(np.arange(n), rows[np.isin(les, ids) & (fracs > 0)])
        full = np.ones(n, bool)
        lesion_any = any_patch(np.arange(len(diam)))
        p = {"p_full": prob(model, h, full), "p_topk": prob(model, h, top),
             "p_minus_lost": prob(model, h, ~any_patch(lost_ids)) if lost_ids else float("nan"),
             "p_minus_kept": prob(model, h, ~any_patch(kept_ids)) if kept_ids and (~any_patch(kept_ids)).any() else float("nan"),
             "p_minus_all": prob(model, h, ~lesion_any) if (~lesion_any).any() else float("nan")}
        classes = [size_class(dm / 1000, strata["itc_max"], strata["micro_max"]) for dm in diam]
        stratum = "macro" if "macro" in classes else "micro" if "micro" in classes else "itc"
        slide_rows.append({"dataset": dataset, "fold": fold, "slide_id": s.slide_id, "stratum": stratum,
                           "n_patches": n, "n_lost": len(lost_ids), "n_kept": len(kept_ids),
                           "topk_attn_share": float(a[top].sum()), "threshold": t, **p,
                           **{k.replace("p_", "call_"): int(v > t) if v == v else -1 for k, v in p.items()}})
    return lesion_rows, slide_rows


def summarise(L, S):
    lines = ["# Do the lesions Top-K ABMIL drops matter to the full-bag decision? (K = 32)", "",
             "Same model: Top-K ABMIL = full-bag ABMIL restricted to its own top-K attention patches. "
             "Pooled over 5 fold models. Contribution = share of the full-bag slide logit difference "
             "(exact, linear classifier). Calls at the fold's CAMELYON16 threshold.", ""]
    for ds in sorted({r["dataset"] for r in L}):
        lines += [f"## {ds}", "", "### Lesions (hittable at the primary threshold)", "",
                  "| size | kept/lost | n | median attention mass | median contribution share | median best rank |",
                  "|---|---|---|---|---|---|"]
        for z in ("itc", "micro", "macro"):
            for k, lab in ((1, "kept"), (0, "lost")):
                rs = [r for r in L if r["dataset"] == ds and r["size_class"] == z and r["kept"] == k]
                if not rs:
                    continue
                share = [r["contrib"] / r["slide_logit_diff"] if r["slide_logit_diff"] > 0 else np.nan for r in rs]
                lines.append(f"| {z} | {lab} | {len(rs)} | {np.median([r['attn_mass'] for r in rs]):.4f} | "
                             f"{np.nanmedian(share):.4f} | {int(np.median([r['best_rank'] for r in rs]))} |")
        lines += ["", "### Slides", "",
                  "| stratum | n (slide x fold) | top-K attention share (median) | called: full | top-K | "
                  "full minus lost lesions | full minus kept lesions | full minus all lesions | "
                  "call flips when lost lesions removed |", "|---|---|---|---|---|---|---|---|---|"]
        for st in ("micro", "macro", "itc"):
            rs = [r for r in S if r["dataset"] == ds and r["stratum"] == st]
            if not rs:
                continue
            frac = lambda c: np.mean([r[c] for r in rs if r[c] >= 0]) if any(r[c] >= 0 for r in rs) else np.nan
            withlost = [r for r in rs if r["call_minus_lost"] >= 0]
            flips = sum(r["call_full"] != r["call_minus_lost"] for r in withlost)
            lines.append(f"| {st} | {len(rs)} | {np.median([r['topk_attn_share'] for r in rs]):.2f} | "
                         f"{frac('call_full'):.2f} | {frac('call_topk'):.2f} | {frac('call_minus_lost'):.2f} | "
                         f"{frac('call_minus_kept'):.2f} | {frac('call_minus_all'):.2f} | "
                         f"{flips}/{len(withlost)} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c16-config", required=True)
    ap.add_argument("--c17-config", required=True)
    ap.add_argument("--backbone", default="uni")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, tgt = load_config(args.c16_config), load_config(args.c17_config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    primary = src["primary_overlap_threshold"]
    runs = Path(src["paths"]["output_root"]) / "runs" / args.backbone / "abmil_full"

    c16, c17 = src["paths"]["cache_root"], tgt["paths"]["cache_root"]
    todo = [("c16_test", c16, i) for i in slide_ids(c16, args.backbone) if i.startswith("test_")]
    todo += [("c17_annotated", c17, i) for i in slide_ids(c17, args.backbone)]
    models, thr = [], []
    for fold in range(src["training"]["n_folds"]):
        d = runs / f"fold{fold}" / "Kfull"
        state = torch.load(d / "model.pt", map_location="cpu")
        model = GatedABMIL(state["fc.0.weight"].shape[1] if "fc.0.weight" in state else 1024,
                           dropout=src["training"]["dropout"])
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        normals = [float(r["prob"]) for r in read_csv(d / "predictions.csv") if r["split"] == "val" and r["label"] == "0"]
        thr.append(spec_threshold(normals, SPEC))
    L, S = [], []
    for j, (ds, cache, sid) in enumerate(todo):
        s = load_slide(cache, args.backbone, sid, label_threshold=primary)
        if not len(s.lesion_diameter_um):
            continue
        for fold, (model, t) in enumerate(zip(models, thr)):
            l, r = analyse(ds, s, model, args.K, t, primary, src["strata_mm"], fold)
            L += l
            S += r
        print(f"{j + 1}/{len(todo)} {sid}", flush=True)
    write_csv(out / "lesions.csv", L)
    write_csv(out / "slides.csv", S)
    (out / "summary.md").write_text(summarise(L, S))
    print(out / "summary.md")


if __name__ == "__main__":
    main()
