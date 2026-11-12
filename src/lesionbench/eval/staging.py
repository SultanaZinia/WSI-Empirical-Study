"""pN staging on CAMELYON17 from each method's kept patches (clinical consequence of patch selection).

Per slide a 4-class call (negative / itc / micro / macro); per patient (5 nodes) the pN stage by the
CAMELYON17 challenge rules; scored by quadratic weighted kappa, as in the challenge evaluation.
"""

import numpy as np
import torch

from ..selectors.abmil import GatedABMIL
from ..train import train_mil

SLIDE_CLASSES = ["negative", "itc", "micro", "macro"]
PN_STAGES = ["pN0", "pN0(i+)", "pN1mi", "pN1", "pN2"]


def pn_stage(slide_stages: list[str]) -> str:
    """CAMELYON17 rules: positive node = micro or macro (ITC does not count).
    No positive node: pN0(i+) if any ITC else pN0; positive nodes but no macro: pN1mi;
    with a macro: pN1 for 1-3 positive nodes, pN2 for 4 or more."""
    n_pos = sum(s in ("micro", "macro") for s in slide_stages)
    if n_pos == 0:
        return "pN0(i+)" if "itc" in slide_stages else "pN0"
    if "macro" not in slide_stages:
        return "pN1mi"
    return "pN1" if n_pos <= 3 else "pN2"


def quadratic_kappa(y_true: list[str], y_pred: list[str], labels: list[str] = PN_STAGES) -> float:
    """Cohen's kappa with quadratic weights over the ordered ``labels``."""
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    obs = np.zeros((n, n))
    for t, p in zip(y_true, y_pred, strict=True):
        obs[idx[t], idx[p]] += 1
    w = (np.arange(n)[:, None] - np.arange(n)[None, :]) ** 2 / (n - 1) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    return float(1 - (w * obs).sum() / (w * exp).sum())


def patient_folds(patient_stage: dict[str, str], n_folds: int, seed: int) -> dict[str, int]:
    """Patient -> fold, stratified by pN stage: patients of each stage are shuffled (seeded) and dealt round
    robin, continuing the deal across stages so fold sizes differ by at most one."""
    rng = np.random.default_rng(seed)
    out, k = {}, 0
    for stage in PN_STAGES:
        ps = sorted(p for p, s in patient_stage.items() if s == stage)
        for p in rng.permutation(ps).tolist():
            out[p] = k % n_folds
            k += 1
    return out


def cross_validate(bags: dict, slide_stage: dict[str, str], patient_of: dict[str, str], folds: dict[str, int],
                   tcfg: dict, seed: int, device="cpu", dropout: float = 0.25) -> list[dict]:
    """Out-of-fold 4-class slide calls. ``bags``: slide -> kept features (n, D). For test fold c, the head is
    trained on the other folds except c + 1 (mod n), which is its early-stopping validation fold; seed + c, so
    every method sees the same folds and initial weights. Returns one row per slide, with the round's epochs
    run and best epoch (the restored weights)."""
    n_folds = max(folds.values()) + 1
    cls = {c: i for i, c in enumerate(SLIDE_CLASSES)}
    in_dim = next(iter(bags.values())).shape[1]
    rows = []
    for c in range(n_folds):
        def part(k):
            return [s for s in sorted(bags) if folds[patient_of[s]] == k]
        test, val = part(c), part((c + 1) % n_folds)
        train = [s for s in sorted(bags) if folds[patient_of[s]] not in (c, (c + 1) % n_folds)]
        model, hist = train_mil(lambda: GatedABMIL(in_dim, dropout=dropout, n_classes=len(SLIDE_CLASSES)),
                                [(bags[s], cls[slide_stage[s]]) for s in train],
                                [(bags[s], cls[slide_stage[s]]) for s in val], tcfg, seed + c, device=device)
        val_losses = [h["val_loss"] for h in hist]
        best_epoch = int(np.argmin(val_losses))
        model.eval()
        with torch.no_grad():
            for s in test:
                p = torch.softmax(model(bags[s].to(device))[0], dim=0).cpu().numpy()
                rows.append({"slide_id": s, "patient": patient_of[s], "cv_fold": c, "true": slide_stage[s],
                             "pred": SLIDE_CLASSES[int(p.argmax())], "epochs": len(hist),
                             "best_epoch": best_epoch, "best_val_loss": float(val_losses[best_epoch]),
                             **{f"p_{k}": float(v) for k, v in zip(SLIDE_CLASSES, p, strict=True)}})
    return rows


def patient_rows(slide_rows: list[dict], patient_stage: dict[str, str]) -> list[dict]:
    """pN stage per patient from the predicted slide classes (true = the official pN label)."""
    by = {}
    for r in slide_rows:
        by.setdefault(r["patient"], []).append(r["pred"])
    return [{"patient": p, "true": patient_stage[p], "pred": pn_stage(v), "n_slides": len(v)}
            for p, v in sorted(by.items())]
