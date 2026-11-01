import math

import pytest
import torch

from lesionbench.metrics.retention import lsr_curve
from lesionbench.selectors.base import topk_indices
from lesionbench.selectors.controls import OracleLSRSelector, OracleSelector, RandomSelector, random_hit_prob

F = torch.zeros(5, 3)  # features are unused by the controls
LABELS = torch.tensor([0, 1, 0, 1, 1])


def test_topk_ties_go_to_lower_index():
    s = torch.tensor([1.0, 3.0, 3.0, 2.0, 3.0])
    assert topk_indices(s, 2).tolist() == [1, 2]
    assert topk_indices(s, 4).tolist() == [1, 2, 4, 3]
    assert topk_indices(s, 10).tolist() == [1, 2, 4, 3, 0]  # K > N -> all rows


def test_random_selector_deterministic_unique_and_k_gt_n():
    sel = RandomSelector()
    a = sel.select(F, 3, seed=7).retained_idx
    assert a.tolist() == sel.select(F, 3, seed=7).retained_idx.tolist()
    assert len(set(a.tolist())) == 3 and a.min() >= 0 and a.max() < 5
    assert sorted(sel.select(F, 10, seed=7).retained_idx.tolist()) == [0, 1, 2, 3, 4]
    picks = {tuple(sel.select(F, 3, seed=s).retained_idx.tolist()) for s in range(20)}
    assert len(picks) > 1


def test_oracle_positives_first_then_fill():
    sel = OracleSelector()
    two = sel.select(F, 2, patch_labels=LABELS, seed=0).retained_idx.tolist()
    assert len(two) == 2 and set(two) <= {1, 3, 4}  # K < n_pos: only positives
    four = sel.select(F, 4, patch_labels=LABELS, seed=0).retained_idx.tolist()
    assert set(four[:3]) == {1, 3, 4} and four[3] in {0, 2}  # all positives, then one negative
    assert sorted(sel.select(F, 10, patch_labels=LABELS, seed=0).retained_idx.tolist()) == [0, 1, 2, 3, 4]
    assert two == sel.select(F, 2, patch_labels=LABELS, seed=0).retained_idx.tolist()


def test_oracle_on_negative_slide_is_random_fill():
    idx = OracleSelector().select(F, 3, patch_labels=torch.zeros(5), seed=0).retained_idx.tolist()
    assert len(set(idx)) == 3


# --- Oracle-LSR and exact random hit probability ---------------------------------------------------

# 10 rows. Lesion 0 (big): rows 0-4 at 0.9 .. 0.5. Lesion 1: row 6 (0.3). Lesion 2: row 8 (0.04).
# Row 5 touches lesions 0 (0.2) and 3 (0.2); lesion 3 has nothing else.
OV = (
    [0, 1, 2, 3, 4, 5, 6, 8, 5],
    [0, 0, 0, 0, 0, 0, 1, 2, 3],
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.3, 0.04, 0.2],
)
F10 = torch.zeros(10, 1)


def _olsr(K, t, seed=0, **kw):
    return OracleLSRSelector().select(F10, K, overlap_rows=OV[0], overlap_lesions=OV[1], overlap_fracs=OV[2],
                                      threshold=t, seed=seed, **kw).retained_idx.tolist()


def test_oracle_lsr_reaches_min_k_hittable_at_every_threshold():
    for t, hittable in ((0.0, 4), (0.05, 3), (0.10, 3), (0.25, 2), (0.50, 1)):
        for K in range(1, 11):
            for seed in range(5):
                idx = _olsr(K, t, seed)
                assert len(idx) == len(set(idx)) == K
                lsr = lsr_curve(idx, *OV, 4, [t])[t]
                assert lsr >= min(K, hittable) / 4 - 1e-12, (t, K, seed)


def test_oracle_lsr_first_pass_skips_lesions_already_hit():
    # At 0.10 lesions 0, 1, 3 are hittable. Row 5 hits lesions 0 and 3 together, so whatever the lesion order,
    # K = 3 hits all three and always includes rows 5 and 6.
    for seed in range(10):
        idx = _olsr(3, 0.10, seed)
        assert {5, 6} <= set(idx)
        assert lsr_curve(idx, *OV, 4, [0.10])[0.10] == 0.75


def test_oracle_lsr_priority_puts_chosen_lesions_first():
    # Priority: lesion 2 first, then 1, then the rest. K = 2 must hit lesions 2 and 1.
    idx = _olsr(2, 0.0, lesion_priority=[2, 1, 0, 2])
    assert set(idx) == {8, 6}


def test_oracle_lsr_k_gt_n_and_no_lesions():
    assert sorted(_olsr(20, 0.0)) == list(range(10))
    idx = OracleLSRSelector().select(F10, 3, overlap_rows=[], overlap_lesions=[], overlap_fracs=[],
                                     threshold=0.1, seed=1).retained_idx.tolist()
    assert len(set(idx)) == 3


def test_random_hit_prob_closed_form_and_simulation():
    assert random_hit_prob(10, 3, 0) == 0.0
    assert random_hit_prob(10, 10, 1) == 1.0 and random_hit_prob(10, 8, 3) == 1.0
    assert random_hit_prob(10, 3, 2) == pytest.approx(1 - math.comb(8, 3) / math.comb(10, 3))
    assert random_hit_prob(5000, 32, 3) == pytest.approx(1 - math.comb(4997, 32) / math.comb(5000, 32))
    # Harness sanity check: the simulated Random selector agrees with the formula.
    sel, f40 = RandomSelector(), torch.zeros(40, 1)
    hits = sum(bool({0, 1, 2} & set(sel.select(f40, 5, seed=s).retained_idx.tolist())) for s in range(4000))
    assert hits / 4000 == pytest.approx(random_hit_prob(40, 5, 3), abs=0.03)
