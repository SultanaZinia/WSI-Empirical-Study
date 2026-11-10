"""One recipe per method: fit on a fold's training slides, then select K patches and predict for any slide.

Every slide prediction uses only what the method kept at K (matched budget); ``abmil_full`` (full-bag ABMIL)
is the no-budget reference. With ``state`` (a run's saved model.pt), ``fit`` rebuilds the fitted method from
those weights instead of training it (external validation); K-independent pieces that were not saved (normal
key banks) are rebuilt from the same fold training slides. Fold-level pieces that do not depend on K (ABMIL
model, normal key banks, PMIL head and prototypes) are fitted once per fold and shared through ``cache``. Only
the Oracle reads patch labels.
"""

import copy
import logging
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..selectors.abmil import ABMILTopK, GatedABMIL
from ..selectors.base import SelectionResult, budget_k
from ..selectors.controls import OracleSelector, RandomSelector
from ..selectors.dtfd import DTFD, DTFDSelector, train_dtfd
from ..selectors.lts import LTS, LTSSelector, cosine_tau
from ..selectors.normal_bank import NormalBankSelector, cur_keys, dfi_keys
from ..selectors.pmil import PMIL, PMILSelector, affinity_prototypes, maxmil_pretrain, train_pmil
from ..selectors.slicechat import SLICEChatMIL, SLICEChatSelector, aux_loss
from ..selectors.tcssa import TCSSA, TCSSASelector, train_tcssa
from ..train import train_mil
from .data import Slide

METHODS = ["random", "oracle", "abmil", "siimil", "nrk_abmil", "pmil", "tc_ssa", "lts", "slicechat", "dtfd"]
PERCENT_METHODS = {"random", "oracle", "siimil", "nrk_abmil", "slicechat"}  # budget "pXX" allowed
PMIL_TOP_N = 1000  # patches per training slide clustered into prototypes (official max-mil.py --n)
log = logging.getLogger(__name__)


@dataclass
class Fitted:
    select: Callable[[Slide], SelectionResult]
    prob: Callable[[Slide, SelectionResult], float]  # P(tumour) from the kept patches only
    history: list
    model: torch.nn.Module  # what prob() runs (the K-patch ABMIL head for fixed selectors)


def slide_seed(slide: Slide, seed: int) -> int:
    return zlib.crc32(slide.slide_id.encode()) ^ seed


def _p(logits) -> float:
    return torch.softmax(logits, dim=-1)[1].item()


def _cap(n: int, tcfg: dict) -> int:
    """Epoch count, shortened by ``epoch_cap`` (smoke runs only)."""
    return min(n, tcfg.get("epoch_cap") or n)


def _abmil(bags_tr, bags_va, in_dim, tcfg, seed, device):
    tc = {**tcfg, "max_epochs": _cap(tcfg["max_epochs"], tcfg)}
    return train_mil(lambda: GatedABMIL(in_dim, dropout=tcfg["dropout"]), bags_tr, bags_va, tc, seed,
                     device=device)


def _loaded(model: torch.nn.Module, state: dict, device) -> torch.nn.Module:
    model.load_state_dict(state)
    return model.to(device).eval()


def _fixed_selection(select_fn, train, val, tcfg, seed, device, state=None) -> Fitted:
    """A selector needing no slide-label training (Random, Oracle, normal banks) + ABMIL on its K patches."""
    def bags(slides):
        return [(s.features[select_fn(s).retained_idx], s.label) for s in slides]

    in_dim = train[0].features.shape[1]
    if state is None:
        model, hist = _abmil(bags(train), bags(val), in_dim, tcfg, seed, device)
    else:
        model, hist = _loaded(GatedABMIL(in_dim, dropout=tcfg["dropout"]), state, device), []
    model.eval()

    @torch.no_grad()
    def prob(s, sel):
        return _p(model(s.features[sel.retained_idx].to(device))[0])

    return Fitted(select_fn, prob, hist, model)


def _bank(kind: str, train: list[Slide], seed: int, device) -> torch.Tensor:
    """Normal key bank from the fold's training normal slides only (concatenated per-slide key sets).
    SiiMIL's CUR keys are built on CPU; NRK's de-duplication runs on ``device``."""
    normals = [s for s in train if s.label == 0]
    if kind == "siimil":
        keys = [cur_keys(s.features) for s in normals]
    else:
        keys = [dfi_keys(s.features.to(device), 0.95, seed=slide_seed(s, seed)) for s in normals]
    return torch.cat([k.to(device) for k in keys])


def _pmil_parts(train, tcfg, seed, device):
    in_dim = train[0].features.shape[1]
    head = maxmil_pretrain([(s.features, s.label) for s in train], in_dim, seed, epochs=_cap(30, tcfg),
                           device=device)
    with torch.no_grad():
        tops = []
        for s in train:
            prob = torch.softmax(head(s.features.to(device)), dim=1)[:, 1].cpu()
            tops.append(s.features[torch.argsort(-prob, stable=True)[:PMIL_TOP_N]].numpy())
    protos = torch.from_numpy(affinity_prototypes(tops, seed=24))  # random_state 24 as in cluster.py
    log.info("PMIL: %d prototypes from %d training slides", len(protos), len(train))
    return head, protos


def fit(method: str, K: int | str, train: list[Slide], val: list[Slide], tcfg: dict, seed: int, device,
        cache: dict, state: dict | None = None) -> Fitted:
    in_dim = train[0].features.shape[1]
    if isinstance(K, str) and method not in PERCENT_METHODS:
        raise ValueError(f"{method} has a fixed K budget; percentages are for {sorted(PERCENT_METHODS)}")
    if method == "random":
        return _fixed_selection(lambda s: RandomSelector().select(s.features, budget_k(K, len(s.features)),
                                                                  seed=slide_seed(s, seed)),
                                train, val, tcfg, seed, device, state)
    if method == "oracle":
        return _fixed_selection(
            lambda s: OracleSelector().select(s.features, budget_k(K, len(s.features)),
                                              patch_labels=torch.from_numpy(s.patch_labels),
                                              seed=slide_seed(s, seed)), train, val, tcfg, seed, device,
            state)
    if method in ("siimil", "nrk_abmil"):
        if method not in cache:
            cache[method] = _bank(method, train, seed, device)
        sel = NormalBankSelector(cache[method], 150 if method == "siimil" else 5)
        memo: dict[str, SelectionResult] = {}

        def select(s):
            if s.slide_id not in memo:
                r = sel.select(s.features.to(device), budget_k(K, len(s.features)))
                memo[s.slide_id] = SelectionResult(retained_idx=r.retained_idx.cpu(), scores=r.scores.cpu())
            return memo[s.slide_id]

        return _fixed_selection(select, train, val, tcfg, seed, device, state)
    if method in ("abmil", "abmil_full"):
        if state is not None:
            cache["abmil"] = (_loaded(GatedABMIL(in_dim, dropout=tcfg["dropout"]), state, device), [])
        if "abmil" not in cache:
            cache["abmil"] = _abmil([(s.features, s.label) for s in train],
                                    [(s.features, s.label) for s in val], in_dim, tcfg, seed, device)
        model, hist = cache["abmil"]
        model.eval()
        selector = ABMILTopK(model)

        def select(s):
            r = selector.select(s.features.to(device), K if method == "abmil" else len(s.features))
            return SelectionResult(retained_idx=r.retained_idx.cpu(), scores=r.scores.cpu())

        @torch.no_grad()
        def prob(s, r):
            return _p(model(s.features[r.retained_idx].to(device))[0])

        return Fitted(select, prob, hist, model)
    if method == "pmil":
        if state is not None:
            model = PMIL(torch.nn.Linear(in_dim, 2), torch.zeros_like(state["prototypes"]), s=K)
            model, hist = _loaded(model, state, device), []
        else:
            if "pmil" not in cache:
                cache["pmil"] = _pmil_parts(train, tcfg, seed, device)
            head, protos = cache["pmil"]
            model, hist = train_pmil(lambda: PMIL(copy.deepcopy(head), protos, s=K),
                                     [(s.features, s.label) for s in train],
                                     [(s.features, s.label) for s in val], seed, epochs=_cap(30, tcfg),
                                     device=device)
        model.eval()

        def select(s):
            r = PMILSelector(model.head).select(s.features.to(device), K)
            return SelectionResult(retained_idx=r.retained_idx.cpu(), scores=r.scores.cpu())

        @torch.no_grad()
        def prob(s, r):  # PMIL's bag head codes exactly these top-K patches
            return _p(model(s.features.to(device))[1])

        return Fitted(select, prob, hist, model)
    if method == "tc_ssa":
        if state is not None:
            model, hist = _loaded(TCSSA(in_dim, n_slots=K), state, device), []
        else:
            model, hist = train_tcssa(lambda: TCSSA(in_dim, n_slots=K),
                                      [(s.features, s.label) for s in train],
                                      [(s.features, s.label) for s in val], seed, max_epochs=_cap(50, tcfg),
                                      device=device)
        model.eval()

        def select(s):
            r = TCSSASelector(model).select(s.features.to(device), K)
            return SelectionResult(retained_idx=r.retained_idx, W=r.W.cpu())

        @torch.no_grad()
        def prob(s, r):  # classified from the K slots
            return _p(model(s.features.to(device))[0])

        return Fitted(select, prob, hist, model)
    if method == "lts":
        tc = {**tcfg, "max_epochs": _cap(tcfg["max_epochs"], tcfg)}

        def anneal(m, epoch):
            m.tau = cosine_tau(epoch, anneal_epochs=tcfg["stop_epoch"])

        if state is not None:
            model, hist = _loaded(LTS(in_dim, K, dropout=tcfg["dropout"]), state, device), []
        else:
            model, hist = train_mil(lambda: LTS(in_dim, K, dropout=tcfg["dropout"]),
                                    [(s.features, s.label) for s in train],
                                    [(s.features, s.label) for s in val], tc, seed, on_epoch=anneal,
                                    device=device)
        model.eval()

        def select(s):
            r = LTSSelector(model).select(s.features.to(device), K)
            return SelectionResult(retained_idx=r.retained_idx.cpu(), scores=r.scores.cpu())

        @torch.no_grad()
        def prob(s, r):  # eval forward keeps the same hard top-K
            return _p(model(s.features.to(device))[0])

        return Fitted(select, prob, hist, model)
    if method == "slicechat":
        tc = {**tcfg, "max_epochs": _cap(tcfg["max_epochs"], tcfg)}
        if state is not None:
            model, hist = _loaded(SLICEChatMIL(in_dim, K, dropout=tcfg["dropout"]), state, device), []
        else:
            model, hist = train_mil(lambda: SLICEChatMIL(in_dim, K, dropout=tcfg["dropout"]),
                                    [((s.features, s.grid), s.label) for s in train],
                                    [((s.features, s.grid), s.label) for s in val], tc, seed,
                                    extra_loss=aux_loss, device=device)
        model.eval()

        def select(s):
            r = SLICEChatSelector(model).select(s.features.to(device), K, grid=s.grid.to(device))
            return SelectionResult(retained_idx=r.retained_idx.cpu(), layer_of_drop=r.layer_of_drop.cpu(),
                                   stage_scores=r.stage_scores.cpu())

        @torch.no_grad()
        def prob(s, r):
            return _p(model((s.features.to(device), s.grid.to(device)))[0])

        return Fitted(select, prob, hist, model)
    if method == "dtfd":
        if state is not None:
            model, hist = _loaded(DTFD(in_dim), state, device), []
        else:
            model, hist = train_dtfd(lambda: DTFD(in_dim), [(s.features, s.label) for s in train],
                                     [(s.features, s.label, slide_seed(s, seed)) for s in val], K, seed,
                                     epochs=_cap(200, tcfg), device=device)
        model.eval()

        def select(s):  # one pseudo-bag split per slide, seeded
            r = DTFDSelector(model).select(s.features.to(device), K, seed=slide_seed(s, seed))
            return SelectionResult(retained_idx=r.retained_idx.cpu(), scores=r.scores.cpu())

        @torch.no_grad()
        def prob(s, r):  # Tier 2 on the kept patches only
            return _p(model.predict_kept(s.features[r.retained_idx].to(device)))

        return Fitted(select, prob, hist, model)
    raise ValueError(f"unknown method {method!r}")


