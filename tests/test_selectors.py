import torch

from lesionbench.selectors.base import topk_indices
from lesionbench.selectors.controls import OracleSelector, RandomSelector

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
