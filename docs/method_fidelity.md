# Method fidelity audit (2026-11-05)

Each method compared with its paper and official code (cloned read-only on 2026-11-05; commit given).
Details of every change are in `docs/deviations.md`. Verdicts: **Faithful** (official model and training
path, only shared-setup changes), **Adapted** (core mechanism as in the code, setting or training changed),
**Re-implemented** (no official code; built from the paper text).

Shared changes for every method (design of the study, not per-method choices): frozen UNI features
(CONCH second), slide labels only, exactly K kept, K in {32, 64, 128, 256}.

| Method | Source | Mechanism matches code? | Main changes | Designed budget | Our K = 32 vs designed | Verdict |
|---|---|---|---|---|---|---|
| Top-K ABMIL | Ilse et al. 2018; CLAM recipe | Yes (gated ABMIL, CLAM "small") | Top-K by attention after full-bag training (post-hoc) | Full bag (not a compressor) | Not applicable (our construction) | Faithful model; budget is ours |
| SiiMIL | Su et al. 2023; JoeSu666/SiiMIL @ 8c140df | Yes: CUR normal keys, mean top-150 cosine, least-normal first (`sii.py`, `mydataset.py`) | Top-K instead of top fraction; shared gated-ABMIL head (official: same gated ABMIL 1024→512→256, 1 logit + BCE) | **30 %** of patches (`train_cv.py -r 0.3`) | ~0.6 % of patches: **~50× fewer** | Adapted |
| NRK-ABMIL | Sajjad et al. 2023; cialab/NRKMIL @ 1b8b144 | Yes: DFI de-duplication at 0.95, mean top-5 cosine | Top-K instead of top fraction; per-slide banks concatenated (pooling step not in repo); shared head (official: same gated ABMIL) | **10 %** of patches (`'r': 0.10`) | **~16× fewer** | Adapted |
| PMIL | Yu et al. 2023; zihou98/PMIL @ f9b241a | Yes: max-MIL instance head, AP prototypes, normalised Euclidean coding, bag + instance loss | Linear head on frozen features instead of fine-tuned ResNet-34; no validation slides in max-MIL; s = K | **s = 5** patches (hard-coded `m_indices[:5]`) | **More** than designed (32 vs 5) | Adapted |
| TC-SSA | Chen et al. 2026; OzzyChen97/TC-SSA @ fbadc98 | Yes, line by line (`moe_compressor.py`, `wsi_classifier.py`, `tools/train.py` defaults) | No AMP; slide-classification path, not the VQA/LLM path | **32** slots default (configs 32–128), UNI 1024-d | **Same** | Faithful |
| LTS | Chen et al. arXiv 2606.08641; no code | Not checkable | Scorer size, Φ_K, τ schedule, denoiser width guessed; LLM replaced by gated-ABMIL head | **K = 32** (only budget in paper) | **Same** | Re-implemented |
| SLICEChat-style | Bozkurt et al. (WACV 2027); ali-kerem/SLICEChat-encoder @ c665990 | Yes: Cropr pruner (1 query, pre-softmax score, detached aux), Hilbert sort, one contiguous window removed per stage, 5 pruners | No MAE pretraining or CLIP pruner training (joint from scratch, slide labels); d 256 vs 768; "MT" vs "MMMT" × 6; no 2D RoPE; no drop-path | **20 %** kept (`pruning_rate 0.8`; per stage keep 72.5 %) | **~30× fewer** (per stage keep ~36 %, i.e. ~64 % removed as one region) | Adapted, far outside designed budget |

## What this means for the results

- K = 32 is inside the designed budget for TC-SSA and LTS, above it for PMIL, and far below it for
  SiiMIL (~50×), NRK-ABMIL (~16×) and SLICEChat-style (~30×).
- Of the three methods that lose small lesions, SiiMIL and SLICEChat-style run far below their designed
  budget; TC-SSA does not (its loss is dilution at its own default of 32 slots).
- NRK-ABMIL also runs ~16× below its budget but keeps micro lesions (0.90 on C16 micro-only slides), so a
  tight budget alone does not explain failure.
- Suggested check (not yet run): each method at its designed budget (SiiMIL 30 %, NRK 10 %,
  SLICEChat-style 20 %) to separate "fails under a tight budget" from "fails".
