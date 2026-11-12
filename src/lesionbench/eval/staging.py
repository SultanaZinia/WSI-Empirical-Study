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


def _val_metrics(model, val: list[str], bags: dict, slide_stage: dict[str, str], patient_of: dict[str, str],
                 patient_stage: dict[str, str], device) -> dict:
    """Inner-validation metrics of the current weights: cross-entropy (plain and class-balanced: mean of the
    four per-class means), recall per slide class, the two small-lesion errors, false positives on negatives,
    and pN quadratic kappa over the validation patients."""
    cls = {c: i for i, c in enumerate(SLIDE_CLASSES)}
    model.eval()
    ce, pred = {}, {}
    with torch.no_grad():
        for s in val:
            p = torch.softmax(model(bags[s].to(device))[0], dim=0).cpu().numpy()
            ce[s] = float(-np.log(max(p[cls[slide_stage[s]]], 1e-12)))
            pred[s] = SLIDE_CLASSES[int(p.argmax())]
    out = {"val_ce": float(np.mean(list(ce.values())))}
    per_class = []
    for c in SLIDE_CLASSES:
        ss = [s for s in val if slide_stage[s] == c]
        per_class.append(np.mean([ce[s] for s in ss]) if ss else np.nan)
        out[f"recall_{c}"] = float(np.mean([pred[s] == c for s in ss])) if ss else float("nan")
        out[f"n_{c}"] = len(ss)
    out["val_ce_balanced"] = float(np.nanmean(per_class))
    out["itc_to_negative"] = sum(slide_stage[s] == "itc" and pred[s] == "negative" for s in val)
    out["micro_to_negative"] = sum(slide_stage[s] == "micro" and pred[s] == "negative" for s in val)
    out["negative_false_pos"] = sum(slide_stage[s] == "negative" and pred[s] != "negative" for s in val)
    by = {}
    for s in val:
        by.setdefault(patient_of[s], []).append(pred[s])
    pts = sorted(by)
    out["val_pn_kappa"] = quadratic_kappa([patient_stage[p] for p in pts], [pn_stage(by[p]) for p in pts])
    return out


def epoch_curves(bags: dict, slide_stage: dict[str, str], patient_of: dict[str, str], folds: dict[str, int],
                 patient_stage: dict[str, str], tcfg: dict, seed: int, horizon: int, device="cpu",
                 dropout: float = 0.25) -> list[dict]:
    """Diagnostic for checkpoint selection: the same CV rounds and seeds as ``cross_validate``, trained for a
    fixed ``horizon`` (no early stop), with inner-validation metrics (``_val_metrics``) after every epoch.
    Row ``epoch = e`` = weights after e epochs (e = 1..horizon); evaluation uses no random numbers, so the
    training trajectory is the one ``cross_validate`` follows. Test-fold patients are never scored."""
    n_folds = max(folds.values()) + 1
    cls = {c: i for i, c in enumerate(SLIDE_CLASSES)}
    in_dim = next(iter(bags.values())).shape[1]
    fixed = {**tcfg, "max_epochs": horizon + 1, "patience": horizon + 2, "stop_epoch": horizon + 1}
    rows = []
    for c in range(n_folds):
        def part(k):
            return [s for s in sorted(bags) if folds[patient_of[s]] == k]
        val = part((c + 1) % n_folds)
        train = [s for s in sorted(bags) if folds[patient_of[s]] not in (c, (c + 1) % n_folds)]

        def log(model, epoch, c=c, val=val):
            if epoch > 0:
                m = _val_metrics(model, val, bags, slide_stage, patient_of, patient_stage, device)
                rows.append({"cv_fold": c, "epoch": epoch, **m})

        train_mil(lambda: GatedABMIL(in_dim, dropout=dropout, n_classes=len(SLIDE_CLASSES)),
                  [(bags[s], cls[slide_stage[s]]) for s in train],
                  [(bags[s], cls[slide_stage[s]]) for s in val], fixed, seed + c, on_epoch=log, device=device)
    return rows
