import math

import pytest

from lesionbench.eval.external import slide_metrics, spec_threshold


def test_spec_threshold_keeps_at_least_spec_of_normals_below():
    normals = [0.1 * i for i in range(1, 21)]  # 0.1 .. 2.0
    assert spec_threshold(normals, 0.95) == pytest.approx(1.9)  # 19 of 20 <= 1.9
    assert spec_threshold(normals, 0.90) == pytest.approx(1.8)
    assert spec_threshold([0.3], 0.95) == 0.3


def test_slide_metrics_by_stage_and_group_by_hand():
    rows = [{"slide_id": s, "prob": p} for s, p in
            [("n1", 0.1), ("n2", 0.6), ("i1", 0.4), ("i2", 0.7), ("m1", 0.9), ("M1", 0.8)]]
    stage = {"n1": "negative", "n2": "negative", "i1": "itc", "i2": "itc", "m1": "micro", "M1": "macro"}
    centre = {"n1": 0, "n2": 1, "i1": 0, "i2": 1, "m1": 0, "M1": 1}
    out = {r["group"]: r for r in slide_metrics(rows, 0.5, stage, centre)}
    a = out["all"]
    assert a["n"] == 6 and a["specificity"] == 0.5  # n2 (0.6) is above the threshold
    assert a["sens_itc"] == 0.5 and a["sens_micro"] == 1.0 and a["sens_macro"] == 1.0
    # AUC: 4 tumour x 2 normal pairs; only (i1 0.4 < n2 0.6) is misordered -> 7/8
    assert a["auc"] == pytest.approx(7 / 8)
    c0 = out["0"]  # n1, i1, m1
    assert c0["n"] == 3 and c0["sens_itc"] == 0.0 and c0["n_macro"] == 0 and math.isnan(c0["sens_macro"])
    assert c0["auc"] == 1.0
