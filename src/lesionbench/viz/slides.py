"""Helpers for slide-level figures: rule-based slide choice and per-patch values to colour by."""

import numpy as np


def _closest_to_median(rows: list[dict]) -> str:
    n = np.array([r["n_patches"] for r in rows], dtype=float)
    med = np.median(n)
    return min(rows, key=lambda r: (abs(r["n_patches"] - med), r["slide_id"]))["slide_id"]


def choose_slides(info: list[dict]) -> dict[str, str]:
    """Slides for qualitative figures, by a fixed rule (no browsing): the micro slide nearest the median
    patch count, the micro slide with the most micro lesions, and the macro and negative slides nearest the
    median patch count. ``info``: dicts with slide_id, stratum, n_patches, n_micro. Ties: slide id order."""
    by = {s: [r for r in info if r["stratum"] == s] for s in ("micro", "macro", "negative")}
    out = {
        "micro_median": _closest_to_median(by["micro"]),
        "micro_most_lesions": min(by["micro"], key=lambda r: (-r["n_micro"], r["slide_id"]))["slide_id"],
        "macro_median": _closest_to_median(by["macro"]),
        "negative_median": _closest_to_median(by["negative"]),
    }
    return out


def slot_tumour_fraction(
    k: np.ndarray, i: np.ndarray, w: np.ndarray, labels: np.ndarray, n: int
) -> np.ndarray:
    """(n,) for a merging arm with sparse W (slot k, patch i, weight w): the tumour share of the slot each
    patch contributes to most, sum_{tumour j} W_kj / sum_j W_kj. NaN for patches in no slot. 0 = merged into a
    pure normal slot, 1 = a pure tumour slot; tumour patches with low values were diluted."""
    k, i, w = np.asarray(k), np.asarray(i), np.asarray(w, dtype=float)
    n_slots = int(k.max()) + 1 if len(k) else 0
    tot, tum = np.zeros(n_slots), np.zeros(n_slots)
    np.add.at(tot, k, w)
    np.add.at(tum, k, w * labels[i])
    frac = np.divide(tum, tot, out=np.full(n_slots, np.nan), where=tot > 0)
    out = np.full(n, np.nan)
    best = np.full(n, -np.inf)
    for kk, ii, ww in zip(k, i, w, strict=True):
        if ww > best[ii]:
            best[ii], out[ii] = ww, frac[kk]
    return out
