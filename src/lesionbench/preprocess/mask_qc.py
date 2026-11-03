"""Score a candidate tissue mask against the annotations on the existing patch lattice."""

import numpy as np


def score_mask(coords_old, cells, frac, rows, les, ov, diam_um, t_primary, cuts):
    """One dict per tissue cut, plus the lesions lost, for a candidate mask.

    ``cells`` / ``frac``: every lattice cell and its tissue fraction under the candidate
    (``grid.cell_tissue_fraction``). ``coords_old``: the current grid, a subset of ``cells``.
    ``rows``, ``les``, ``ov``: the /overlap (patch row, lesion, fraction) pairs of the current grid.
    A lesion is hittable if some patch overlaps it by >= ``t_primary``, and lost if no such patch is kept.
    Cells new to the candidate are counted (``n_added``) but carry no labels here.
    """
    pos = {(int(x), int(y)): i for i, (x, y) in enumerate(cells)}
    old_idx = np.array([pos[(int(x), int(y))] for x, y in coords_old], dtype=np.int64)
    hit = ov >= t_primary
    tum_rows = np.unique(rows[hit])
    hittable = np.unique(les[hit])
    out, lost = [], []
    for cut in cuts:
        keep_cell = (frac > 0) & (frac >= cut)
        keep_old = keep_cell[old_idx]
        gone = np.setdiff1d(hittable, np.unique(les[hit & keep_old[rows]]))
        out.append({
            "cut": cut,
            "n_old": len(coords_old),
            "n_new": int(keep_cell.sum()),
            "n_dropped": int((~keep_old).sum()),
            "n_added": int(keep_cell.sum() - keep_old.sum()),
            "tumour_patches": len(tum_rows),
            "tumour_dropped": int((~keep_old[tum_rows]).sum()),
            "lesions_hittable": len(hittable),
            "lesions_lost": len(gone),
        })
        lost += [{"cut": cut, "lesion_id": int(g), "diameter_um": float(diam_um[g])} for g in gone]
    return out, lost
