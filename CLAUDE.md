# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

**Token Compression for Rare Lesions. Paper 1: Matched-Budget Empirical Benchmark.**

Research question: when whole-slide-image (WSI) pipelines compress thousands of patch tokens down to a
small budget K, do micro-metastases and isolated tumor cells (ITCs) survive? Slide-level AUC can stay
high while small lesions are silently dropped. This project measures *where* evidence is lost, which can be:

- **Encoder stage:** the frozen foundation model never separated tumor from normal.
- **Selector stage:** the encoder separated them, but the selector, pruner or merger discarded or diluted them.

Every method is compared under the **same token budget K ∈ {32, 64, 128, 256}**, on **identical cached features**.

## Pipeline overview

```
CAMELYON16 WSIs + ASAP XML
  → tissue mask (Otsu) + patch grid (256×256 @ 20×, 0.5 µm/px)
  → patch labels + lesion_ids from annotation polygons
  → frozen feature cache (UNI primary, CONCH secondary) in per-slide HDF5
  → ESI audit (encoder separability on micro-metastasis slides)
  → selectors / pruners / mergers at matched K, with token lineage tracking
  → evidence-retention metrics + slide-level metrics, stratified by lesion size
```

## Data

- **Dataset:** official CAMELYON16, 270 train WSIs and 129 test WSIs.
- **Test strata.** Compute these from the annotations. Do not hard-code them:
  - Negative: no tumor (expected n = 80)
  - Macro-metastasis: largest contiguous lesion > 2.0 mm (expected n ≈ 28)
  - Micro-metastasis: largest lesion 0.2–2.0 mm (expected n ≈ 18)
  - ITC: lesion < 0.2 mm or a single-cell cluster (expected n ≈ 3)

  The expected counts come from the planning document. If the computed counts differ, report the difference rather than forcing a match.
- **Training:** 5-fold CV over the 270 training slides, using **slide-level labels only** (weak supervision).
  Patch labels are only for evaluation. They must never reach a model's training loss.
- **Never commit raw WSIs, XMLs, HDF5 caches or model weights.** Put data paths in config files, not in code.

### Preprocessing spec

1. **Tissue mask:** two-level Otsu on the HSV saturation/value channels of a ~1.25× thumbnail. Drop glass and adipose.
2. **Patch grid:** non-overlapping 256×256 patches at 20× equivalent (0.5 µm/px). Store coords as level-0 top-left (x, y).
3. **Annotations:** parse the ASAP XML into `shapely.geometry.Polygon`. Assign each contiguous lesion component
   `lesion_id ∈ {0..M-1}`. Handle ASAP's `_2` exclusion groups (holes) and invalid polygons by applying `buffer(0)`.
4. **Patch label:** `1` if `Area(patch ∩ lesion) / Area(patch) ≥ 0.10`, otherwise `0`.
   `lesion_id = m` when positive, otherwise `-1`. If a patch overlaps several lesions, it takes the one with the largest overlap.
   ITCs can fall below the 10% threshold. Log every lesion that ends up with zero positive patches. Don't drop them silently.

### Cache schema (one `.h5` per slide)

| Key            | Shape  | dtype   | Meaning                          |
|----------------|--------|---------|----------------------------------|
| `/features`    | (N, D) | float32 | frozen embeddings (UNI D=1024, CONCH D=512) |
| `/coords`      | (N, 2) | int64   | level-0 top-left (x, y)          |
| `/patch_labels`| (N,)   | uint8   | 0 normal, 1 tumor                |
| `/lesion_ids`  | (N,)   | int32   | -1 normal, ≥0 lesion index       |

Store the slide id, backbone name, magnification and patch size as HDF5 attributes. Row order must be the same
across all keys and must not change after creation, because lineage indices point into these rows.

## Backbones

- **UNI** (ViT-L/16, 1024-d) is primary. **CONCH** (ViT-B/16, 512-d) is secondary.
- Both are gated on Hugging Face and need an access token. Keep them frozen and use `torch.no_grad()` with the official preprocessing transforms.

## Methods (matched-budget harmonization)

| Arm | Method | How to force exactly K tokens |
|-----|--------|-------------------------------|
| MIL selection | Top-K ABMIL (gated attention) | top-K attention scores |
| MIL selection | SiiMIL | top-K by salient-instance / outlier score |
| MIL selection | NRK-ABMIL | top-K by distance from the normal-key bank |
| Token pruning | LTS (learnable sparsification) | `torch.topk(gate, K)` replaces the learned threshold |
| Token pruning | Agile Pruner (progressive, in-encoder) | layer-wise schedule ending at K tokens |
| Token merging | TC-SSA (semantic slots) | M = K slots, soft assignment `W ∈ R^{K×N}` |
| Token merging | SLICEChat / PMIL | merge to K centroids by cosine similarity |
| Control | Random | K patches sampled uniformly from tissue |
| Control | Oracle | all positives first (up to K), then random negatives |

**Before implementing a named baseline, check the original paper and code.** The planning document came from
an LLM conversation. Some method names, mechanisms or citations may be inaccurate or may not exist as described.
If a method can't be verified, implement it as a clearly labeled generic family representative (for example
"score-threshold pruner" or "soft-slot aggregator") and record that choice in `docs/deviations.md`.

All methods share one interface. Each returns the retained original patch indices; merging methods also return `W`:

```python
class Selector(Protocol):
    def select(self, features: Tensor, K: int, **ctx) -> SelectionResult: ...
# SelectionResult: retained_idx (K,) LongTensor into original rows; scores (N,) optional;
#                  W (K, N) optional for merging; layer_of_drop (N,) optional for in-encoder pruners
```

`TokenLineageTracker` maps every surviving token, or slot, back to its original patch rows and coordinates.
For in-encoder pruning, it records the layer at which each token was dropped.

## Metrics

Evidence retention is computed per slide, then aggregated per stratum.

- **Cov@K** = Σ_{i∈Retained} y_i / Σ_j y_j, the patch-level tumor recall. Undefined on negative slides, so exclude them.
- **LSR@K** = the fraction of lesions with at least one retained patch.
- **TEP@K** = (1/K) Σ_{i∈Retained} y_i, the purity of the retained set.
- **ΔOracle@K** = Cov_Oracle@K − Cov_Method@K.
- **SDI_k** = Σ_{i:y=0} W_{k,i} / (ε + Σ_{i:y=1} W_{k,i}), for merging methods only. High SDI means the tumor signal was diluted.
  For hard-retention metrics on merging methods, treat a patch as retained if it is the argmax contributor to a slot,
  or if `W_{k,i}` exceeds a fixed threshold. Report which rule was used.
- **Layer-of-Death (LoD):** the mean layer at which ground-truth micro-metastasis tokens are first pruned. In-encoder pruners only.
- **ESI** = ‖μ_tumor − μ_normal‖² / (½(σ²_tumor + σ²_normal)) on raw embeddings for each micro-metastasis slide, plus a linear
  SVM patch-AUC. Decision rule: ESI > 1.5 **and** SVM AUC > 0.90 means the encoder captured the lesion, so any miss belongs to the
  selector. Define σ² as the mean per-dimension variance, and keep that definition consistent everywhere.
- **Slide-level:** ROC-AUC (normal vs metastatic). FROC follows the official CAMELYON16 protocol: mean sensitivity at
  0.25, 0.5, 1, 2, 4 and 8 FP/slide. Use the official evaluation logic, including its ITC exclusion rules.
- **Attribution faithfulness:** Spearman correlation between selector scores and LRP relevance.
- **Statistics:** 95% CIs from a non-parametric bootstrap over slides (B = 1000), and paired Wilcoxon signed-rank tests between
  methods at each K. Correct for multiple comparisons (Holm), and report it.

## Expected deliverables

- **Table 1:** method × K → overall slide AUC, plus Cov@K, LSR@K and TEP@K stratified by Macro, Micro and ITC.
- **Fig 1, "trade-off paradox":** x-axis is K; slide AUC against micro-metastasis LSR@K.
- **Fig 2:** retained-patch overlays from ABMIL, LTS and TC-SSA on micro-metastasis polygons.
- **Fig 3, "stage-loss diagnostic":** ESI against ΔOracle for each micro-metastasis slide.

> **The AUC numbers in the planning document (e.g. Random 0.54, ABMIL 0.92, NRK-ABMIL 0.95, Oracle 1.00) are
> illustrative placeholders, not targets or prior results.** Never write them into code, tests, or the paper
> as findings. All reported numbers must come from runs in this repository.

## Where things run

- **This environment is for development only.** Write code, configs and unit tests here, using synthetic data.
  Don't download CAMELYON16, pull UNI/CONCH weights, or run training or inference here.
- **All experiments run on a separate compute server.** That covers preprocessing, feature extraction, training,
  inference and metric aggregation. Code reaches the server through git: push from here, then pull there.
- Never hard-code server paths, hostnames or credentials. Every data, cache, weight and output location comes from a
  config file or environment variable (e.g. `DATA_ROOT`, `CACHE_ROOT`, `OUTPUT_ROOT`, `HF_TOKEN`). Commit an example config
  (`configs/server.example.yaml`); keep the real server config out of git.
- Scripts must be runnable non-interactively, e.g. under a job scheduler or `nohup`. They take a config path, can resume
  partway (skip slides whose `.h5` or result file already exists) and write logs to `OUTPUT_ROOT`.
- Unit tests must pass without GPUs, OpenSlide, or the real dataset. Gate tests that need real data or weights behind a
  pytest marker (e.g. `@pytest.mark.server`), skipped by default.
- Results come back from the server as small artifacts: metric CSVs and JSON, figures, run metadata. Never bring back HDF5 caches or weights.

### Compute server: Fir (Digital Research Alliance of Canada, accessed via a CCDB account)

- **Host:** `fir.alliancecan.ca`, run by Simon Fraser University. According to the Alliance docs, each GPU node has
  4× H100 SXM5 80 GB, 48 cores and about 1 TB RAM, and the maximum job length is 7 days. Check current details at
  https://docs.alliancecan.ca/wiki/Fir before you rely on them.
- **Scheduler is SLURM.** Every job script must pass `--account=<def-PI or rrg-PI>`. That value comes from the
  config or environment (`SLURM_ACCOUNT`); never hard-code it. Request GPUs in the Alliance style, e.g. `--gpus-per-node=h100:1`.
  Keep job scripts in `slurm/`. They are thin wrappers around `scripts/` entry points.
- **Software environment:** `module load StdEnv/2023 python/3.11` (plus `cuda` if needed). Create a virtualenv with
  `virtualenv --no-download`, then install from the Alliance wheelhouse with `pip install --no-index ...`. Keep
  `requirements-fir.txt` limited to packages the wheelhouse has. If you need OpenSlide, check `module spider openslide` first.
- **Compute nodes have no internet access.** Download everything on a login node *before* submitting jobs.
  That includes CAMELYON16, the gated UNI/CONCH weights (into `HF_HOME`) and any pip packages. Jobs then run with
  `HF_HUB_OFFLINE=1`, and code must never try to download anything at runtime.
- **Storage:**
  - Raw WSIs and HDF5 caches go in `$PROJECT` (shared and persistent, but with quota and file-count limits).
  - Large temporary outputs go in `$SCRATCH`, which is purged on a schedule, so never keep the only copy of anything there.
  - Copy each job's working set to the node-local `$SLURM_TMPDIR` for fast I/O.
  - Keep `$HOME` for code and environments only.
  - Prefer a few large files to many small ones: one `.h5` per slide is fine, but per-patch image files are not.
- **Parallelism:** use SLURM job arrays for per-slide work (patching, feature extraction) and for the run grid of
  method × K × fold. Each array task must be idempotent and must skip finished outputs.
- **Getting code to Fir:** `git pull` on a login node. Results come back via `rsync`/`scp` or Globus, small artifacts only.

## Engineering conventions

- Python ≥ 3.10, PyTorch, numpy, h5py, shapely, scikit-learn, scipy. Use OpenSlide for WSI reads.
- The package is `lesionbench`, in a `src/` layout: `src/lesionbench/{preprocess,features,selectors,lineage,metrics,eval,stats,viz}`.
  Entry points live in `scripts/` and are config-driven. `configs/experiment.yaml` holds the experiment settings and is committed.
  `configs/server.yaml` holds Fir paths and the account, and is git-ignored; copy it from `server.example.yaml`.
  SLURM wrappers live in `slurm/`, and the deviation log is `docs/deviations.md`.
- Commands: `pip install -e ".[dev]"`, `pytest` (skips `server`-marked tests), `ruff check .`.
- **Determinism:** seed everything, including the Random selector, oracle negative fill, CV folds and bootstrap. Log seeds with results.
- **Results provenance:** each run writes its config, git SHA, backbone and seed next to its outputs.
- **Tests:** use pytest on small **synthetic** slides, such as hand-built polygons and fake features, so they run without CAMELYON16 or GPUs.
  Every metric needs hand-computed unit tests, including edge cases: K > N, zero-tumor slides, a lesion with no positive patches, and ties in top-K.
- Patch indices are the shared currency. Never re-sort cached rows. Any selector output must index into the cached order.
- Keep evaluation-only data (`patch_labels`, `lesion_ids`) out of training code paths. The Oracle selector is the only
  sanctioned consumer at selection time.

## Timeline (from plan)

- Weeks 1–2: download, segmentation and patching, label matching, UNI/CONCH feature caching.
- Weeks 3–5: train the baselines (5-fold CV, weak labels) and calibrate budgets; output lineages at each K.
- Week 6: test inference on 129 slides and lineage auditing.
- Weeks 7–8: stratified metrics, bootstrap CIs, Wilcoxon tests, figures.
