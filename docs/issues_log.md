# Issues log

This file records the problems we hit while building the benchmark: methods that didn't match the plan, issues inside official code, gaps in the spec, infrastructure disruptions, and bugs our tests caught. It's meant as source material for the paper's methods and limitations sections.

`docs/deviations.md` has the exact decision for each case, with citations. This log explains what went wrong and why it matters. Last updated 2026-11-01.

## 1. Methods in the plan

The planning document came from an LLM conversation. We checked every named method against its paper and code before implementing it. Several did not hold up.

| Method | Problem | Resolution |
|---|---|---|
| Agile Pruner | **Wrong method.** The real AgilePruner (Baek et al., ICLR 2026, arXiv 2603.01236) prunes tokens for natural-image vision-language models in one shot, using a ViT [CLS] attention map. Cached patch features have no such map. The plan's description ("progressive, in-encoder, layer-wise schedule") actually fits SLICEChat. | Dropped. |
| SLICEChat | **Mislabelled.** The plan listed it as token *merging*. It is *progressive in-encoder pruning* (Bozkurt et al., arXiv 2609.24894). | Moved to the progressive-pruning arm. |
| PMIL | **Ambiguous name.** Two different papers are called PMIL. The topology module (Obeid et al., ISCAS 2025) is not merging. The CAMELYON16, micro-metastasis method is Prototypical MIL (Yu et al., Med Image Anal 2023). A fresh verification found only the first one; the second was found through the literature notes in the separate WSI-experiment repository. | Prototypical MIL (Yu et al. 2023). |
| PTCMIL | Considered as a stand-in merging method. Its repository has **no licence**. | Not used. |
| LTS | **No public code.** The plan said to "replace the learned threshold with top-K", but the method already uses hard top-K at inference. It was tested only at K = 32. | Re-implement from the paper. |
| TC-SSA | The plan described a dense soft assignment W ∈ ℝ^{K×N}; it is **sparse top-2 routing**. The first verification reported only an anonymous code link, but official MIT-licensed code exists (github.com/OzzyChen97/TC-SSA). | Built from the official code. |
| NRK-ABMIL | **A step is missing from the official repo**: the notebooks load a pooled key bank (`train_non_redundant.npy`), but the pooling code is not included. It is also nearly the same method as SiiMIL (same lab; only the key-bank construction differs), so expect correlated results. | Concatenate per-slide banks; logged as unverified. |
| SiiMIL, NRK-ABMIL | They keep a **percentage** of patches (30% and ~10%), not exactly K. | Top-K by each method's own score. |
| LTS, TC-SSA, SLICEChat | Originally trained with a **language-model text loss** (slide VQA), not slide labels. | Retrained with slide-label MIL losses. This is an adaptation and must be stated. |
| SLICEChat | Tied to CONCH v1.5 features, its own slide encoder (pretrained by MAE on 11,410 TCGA slides), Mamba GPU kernels, and LLM training. The core encoder and pruner live in a second repo (SLICEChat-encoder). | Smaller SLICEChat-style encoder retrained on UNI with slide labels (user decision 2026-11-01; docs/deviations.md). |

## 2. Problems inside official code

| Where | Problem | Resolution |
|---|---|---|
| PMIL, `max-mil.py` | Also builds a **training** set from the **validation** slides' top patches, which leaks validation data into training. | Not reproduced. |
| PMIL, `cluster.py` | With the official settings (preference 0, damping 0.5), Affinity Propagation **can fail to converge**. scikit-learn then silently returns every point, or none, as a prototype. It also **collapsed to a single prototype** on unimodal test data. Because each distance row is divided by its maximum, one prototype makes every slide's code identical and the classifier can't distinguish slides. | Explicit convergence check, one retry at damping 0.9 (same fixed point), error otherwise; PMIL refuses fewer than 2 prototypes. Check the prototype count on UNI features before the full run (the official release has 8). |
| PMIL | Fine-tunes a ResNet-34 end to end. | Linear instance head on frozen features. |
| NRK-ABMIL | The 0.95 cosine de-duplication threshold was tuned on ResNet/CTransPath features. In UNI's feature space it may remove very few patches, giving a very large key bank. | Check the bank size on UNI features before the full run. |
| All methods | Originally used other encoders (ResNet-50/34, CTransPath, CONCH v1.5). | Run on UNI (primary) and CONCH (secondary). |

## 3. Gaps and errors in the plan and spec

| Issue | Resolution |
|---|---|
| The spec expected ASAP `_2` exclusion groups; the CAMELYON16 XML files use `Tumor` and `Exclusion` (`_0/_1/_2` is CAMELYON17 naming). One `Dot` annotation exists (test_066, group `None`). | Parser accepts both naming schemes; non-polygon annotations are skipped with a warning. |
| Treating each contiguous annotated region as a lesion disagreed with the official `reference.csv` classes on 3 of 49 test tumour slides. | Regions closer than 75 µm are merged, as in the official `evaluate.py`. Test strata now match the reference exactly. |
| **The plan's expected test strata were wrong**: about 80 negative, 28 macro, 18 micro, 3 ITC. | Official counts: 80 negative, 22 macro, 27 micro, 0 ITC-only slides. ITCs are still analysed at lesion level. |
| Lesion diameter was not defined. | Max Feret diameter (AJCC "greatest dimension"). |
| The minimum tissue fraction for keeping a patch was not defined. | Any tissue, following the TRIDENT default. |
| MIL training settings were not given. | CLAM defaults, checked in its source: Adam, lr **1e-4**, weight decay 1e-5, early stopping on validation loss. (A value of 2e-4 was first stated from memory and corrected after checking.) |
| TEP@K is ill-defined when K > N. | Divide by the number retained, min(K, N). |
| **20–34% of lesions get no positive patch** under the 10% overlap rule (mostly small fragments). | Kept in LSR denominators, so no method, Oracle included, reaches LSR = 1. Report the Oracle ceiling next to every method. |
| **The spec's Oracle is not an LSR ceiling.** It takes random positives, which on macro slides fill K from the large lesion and miss satellites (test macro Oracle LSR@32 = 0.41). | A lesion-aware ("random round-robin") Oracle is deferred until the published methods are done. |
| Tissue-mask QC: 3 of 590 ITC lesions fall entirely outside the patch grid (0 micro, 0 macro), while 99.88% of annotated tumour area is covered. Slide-edge lines and specks get patches on near-empty slides. | Reported, not fixed. Revisit if selectors are seen picking debris. |
| 20 training slides are not exhaustively annotated. 13 training slides still disagree with the official class (2 are ITC vs micro, 4 are non-exhaustive, most others are near the 2 mm cutoff). | Reported. Training labels are used only on the evaluation side. |
| CLAUDE.md says Fir compute nodes have no internet, but earlier download jobs on Fir reached the internet. | Not verified; CLAUDE.md left unchanged. |

## 4. Infrastructure disruptions

| Disruption | Workaround / status |
|---|---|
| **No access yet to the gated UNI and CONCH models** on Hugging Face. This blocks feature extraction and every real result after preprocessing. | Pending. The whole pipeline is built and tested end to end on Fir with a dummy encoder. |
| huggingface.co is unreachable from the development Mac. | Query Hugging Face from Fir. |
| **Fir login node login3 has a filesystem fault** (some home-directory files unreadable). A failed setup attempt there also left virtualenv's shared cache half-built. | Work on login1; build the environment with a fresh temporary cache. |
| The Alliance wheelhouse has only a placeholder for `opencv-python-headless`, which broke `pip install -r requirements-fir.txt`. | Removed (not used). |
| The `conch` package (needed for CONCH) is not in the Alliance wheelhouse. | Install on a login node before CONCH extraction. |
| `git push` over HTTPS from the Mac was refused (403): the stored credentials belong to another GitHub account. | Push over SSH as the repository owner. |
| Running code downloaded from GitHub is blocked in the development environment. | Compared our SiiMIL implementation against a hand transcription of the official algorithm instead. Key sets were identical and scores matched to 2e-8. |
| macOS `md5sum` differs from GNU `md5sum`. | Tested the download script on Fir. |
| **`mamba_ssm` (SLICEChat's Mamba kernel) conflicts with our Fir environment.** The wheelhouse's mamba_ssm 2.2.4 pulls torch down to 2.5.1 (our env has 2.14.1), and it fails to import with the wheelhouse's current `transformers` (it imports the removed `GreedySearchDecoderOnlyOutput`). Also, `virtualenv` failed on login2 with a broken shared cache. | A separate environment for SLICEChat runs: `python -m venv`, `pip install --no-index torch mamba_ssm causal_conv1d "transformers<4.45"` (import verified on login2, 2026-11-01). Our code falls back to a PyTorch scan where the kernel is missing. |

## 5. Bugs caught by our own tests (all fixed)

- **Training was not reproducible.** The model was built before the random seed was set, so its initial weights differed between runs. `train_mil` now builds the model after seeding, and a test checks that same-seed runs match exactly.
- **Empty-slot retention bug.** In `retained_from_assignment`, a merging slot that received no patches (an all-zero row of W) counted patch 0 as retained. That would have happened in practice at K = 256. Fixed and tested.
- **LTS NaN gradients.** At low temperature the soft keep weights saturate to exactly 0 or 1, and the gradient of the square roots in the noise gate became inf × 0 = NaN; training silently diverged at epoch 9 on toy data. The weights are now clamped to [1e-6, 1 − 1e-6], with a test.
- **Setup slips.** `module load` was piped into another command, so it ran in a subshell and was lost. Two expected values in tests were miscalculated by hand (the tests were wrong, not the code; confirmed by direct calculation).
