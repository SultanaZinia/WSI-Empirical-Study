# Evaluation metrics (proposed)

**Status:** proposal, revised 2026-11-01 after an external review. Not yet agreed. Once agreed, update CLAUDE.md's
Metrics section and log the changes in `docs/deviations.md`. Sources marked *(verify)* are named from memory and must be
checked against the primary papers before the methods section is written.

The metrics are organised around four questions, asked in order. Each later question is only meaningful for
lesions that passed the earlier one.

| # | Question | Tier | Needs a classifier? |
|---|----------|------|---------------------|
| Q1 | Did the lesion's evidence survive compression? | 1. Retention | No |
| Q2 | Did losing it change the diagnosis? | 2. Diagnostic consequence | Yes |
| Q3 | Where was it lost: encoder, selector or classifier? | 3. Stage attribution | Partly |
| Q4 | Did the model use the evidence it kept? | 4. Evidence use | Yes |

## Definitions used everywhere

- **Overlap** of patch i with lesion ℓ = Area(patch ∩ lesion) / Area(patch), i.e. a fraction of the **patch**
  (a 128 µm patch at 10% overlap holds ≥ 1,638 µm² of tumour).
- **Hit threshold: decided 2026-11-01.** No published work uses a fraction threshold (PCam and Liu et al. 2017 use
  "any tumour pixel in the centre region"; Wang et al. 2016 and DSMIL use "inside a tumour region"), and the 10% in
  CLAUDE.md came from the planning document without a stated reason. So no single primary threshold is chosen yet:
  - **Now:** LSR is reported at every overlap threshold t ∈ {0 (any overlap), 0.05, 0.10, 0.25, 0.50}
    (`lsr_overlap_thresholds`), from the stored (patch, lesion, overlap) pairs (`/overlap` in the patch files).
    Conclusions are stated only where they hold across the thresholds.
  - **When UNI features exist:** measure the tumour fraction at which the probe reliably detects tumour on
    training-fold slides, and fix the primary threshold there, **before** looking at any selector result. Then the
    same threshold defines y_i, H_ℓ, the probe's labels and TSS (one rule throughout).
  - Until then, `patch_labels` / `lesion_ids` stay at 10% (Cov, TEP, Oracle-Cov use them), as a placeholder.
- **H_ℓ** = tumour patches of lesion ℓ (overlap ≥ 0.10); n_ℓ = |H_ℓ|. Lesions with n_ℓ = 0 cannot be hit by any arm;
  they stay in every LSR denominator, are counted, and are reported as a separate "unhittable" group.
- **Lesion** = connected component of (Tumor − Exclusion) after merging parts closer than 75 µm, as in the official
  CAMELYON16 evaluation. **Size** = max Feret diameter d_ℓ of the undilated lesion.
- **Size classes (AJCC):** ITC d ≤ 0.2 mm; micro 0.2 < d ≤ 2.0 mm; macro d > 2.0 mm.
  Slide stratum = class of its largest lesion (as in pN staging).
- **Slides not exhaustively annotated** (listed in `docs/issues_log.md`) are excluded from lesion-level metrics, because
  unannotated tumour corrupts both hits and misses. They stay in slide-level AUC.
- Slide: N candidate patches, retained set S, |S| = K. Merging arms: assignment matrix W ∈ R^{K×N}.
- Report K/N next to every K, and stratify key results by slide size (terciles of N), since K = 32 is a very different
  compression ratio on small and large slides.

**Other sensitivity analyses for the hit rule:** the centre-region rule of PCam / Liu et al. (any tumour in the
central 128×128 of a 256 patch), and annotation polygons eroded and dilated by a fixed margin chosen before any run
(e.g. 10 µm), to show results don't hinge on how tightly the polygons were drawn.

---

## Q1 — Did the evidence survive? (Tier 1, classifier-free)

| Metric | Definition | Role | Source |
|--------|-----------|------|--------|
| **LSR@K** | lesion-pooled: Σ_ℓ 1[H_ℓ ∩ S ≠ ∅] / #lesions, within a size class | **Primary** | Lesion-wise detection sensitivity (LUNA16; BraTS 2023 lesion-wise metrics) *(verify)* |
| LSR@K, per-slide mean | mean over slides of each slide's LSR | Sensitivity check: pooling weights lesion-rich slides more | — |
| **D50** | diameter where retention crosses 0.5, from an isotonic (monotone) fit of retained vs log d_ℓ, linearly interpolated. Reported next to Random's D50 at the same K, and repeated with n_ℓ in place of d_ℓ. ITCs included | Size-resolved summary; lower is better | Limit-of-detection curves (assay validation) |
| First-hit rank r_ℓ | rank of lesion ℓ's best-ranked patch in the arm's full ordering; LSR@K = P(r_ℓ ≤ K) | **Only for arms with one fixed global ranking** (Random, Top-K ABMIL, SiiMIL, NRK-ABMIL). Not for per-K-trained, clustering or merging arms | — |
| Retention AUC | area under LSR vs log₂ K over the fixed grid {32, 64, 128, 256}, divided by the log range (3) | One number across budgets; comparable only on this grid | CPM-style averaging (LUNA16) *(verify)* |
| Cov@K | Σ_{i∈S} y_i / Σ_j y_j (tumour slides only) | Secondary; dominated by macro lesions | Patch-level retention rate (MedTokenBudget, dermoscopy, 16×16 ViT grid) |
| TEP@K | Σ_{i∈S} y_i / min(K, N) | **Diagnostic only, not a target**: all 32 patches in one macro lesion gives TEP = 1 and poor LSR | — |
| ΔOracle-LSR@K | LSR_Oracle-LSR − LSR_arm | Selector loss on lesions | — |
| ΔOracle-Cov@K | Cov_Oracle-Cov − Cov_arm | Selector loss on area | — |
| NRS@K | (LSR_arm − LSR_Random) / (LSR_Oracle-LSR − LSR_Random), computed from **aggregate rates** within a class, not as a mean of per-lesion ratios | Comparable across K: 0 = random, 1 = oracle | Normalised gain |
| TSS_ℓ (merging only) | max_k Σ_{i∈H_ℓ} f_iℓ W_{k,i} / Σ_i W_{k,i}, with f_iℓ = overlap of patch i with ℓ | Separates "kept" from "kept but diluted"; tumour-fraction weighting stops slots made of edge patches scoring 1 | Partial-volume measures |

**Reference arms for Q1**

- **Random:** use the exact expectation, not sampled seeds. For a lesion with n_ℓ hittable patches,
  P(hit) = 1 − C(N − n_ℓ, K) / C(N, K). Lesion-pooled LSR is the mean of these. (A 20-seed simulation is kept only as a
  unit-test check of the formula.)
- **Oracle-Cov:** tumour patches first, by overlap descending, then random negatives. Ceiling for Cov@K.
- **Oracle-LSR:** round-robin over lesions (each lesion's best patch, then each lesion's second, …), then random
  negatives. Ceiling for overall LSR@K when lesions don't share patches; a patch shared by two lesions counts for both,
  so the ceiling can only rise.
- **Oracle-LSR-micro:** the same round-robin with micro lesions first, then ITC, then macro. The ceiling for **micro**
  LSR, used for H1 and for micro NRS.

**Merging arms, hard retention:** a patch counts as retained if it is the argmax contributor to some slot.
Appendix: W_{k,i} > τ_W with τ_W fixed in advance.

**Scope limit — arms that mix tokens.** In the in-encoder pruner and the merging arms, kept tokens have already
attended to, or been averaged with, lesion tokens. Lesion evidence can survive without its patch, so patch-identity
retention can **undercount** survival for these arms (and the Q4 deletion test can understate use). State this in the
paper. Identity-free check: for every arm, train the Q3 probe's counterpart on the arm's output tokens (training folds)
and report whether the kept tokens of a slide carry lesion signal for lesions that were not retained by identity.

---

## Q2 — Did it change the diagnosis? (Tier 2)

| Metric | Definition | Role | Source |
|--------|-----------|------|--------|
| Slide AUC | ROC-AUC, normal vs metastatic | Shows the dissociation; not evidence on its own | Standard |
| **Stratum sensitivity** | sensitivity on macro / micro slides at the threshold giving 95% specificity on the inner validation split. Report the bootstrap spread of that threshold | Shows failures AUC hides | Screening mammography reporting |
| **Localised sensitivity** | a tumour slide counts only if called positive **and** at least one of the head's top-k attended retained tokens (k fixed in advance, e.g. 5) contains a tumour patch (for slots: its argmax contributor). Protocol C only, where every arm has the same gated-ABMIL head | "Detected for the right reason". One operating point, not an LROC curve; attention is a proxy for importance | LROC idea (Starr/Metz; Swensson) *(verify)* |
| Joint outcome table | per tumour slide: {any lesion retained, none} × {detected, missed}; single-lesion slides reported separately | Links Q1 to Q2 without assuming causality | — |
| FROC | official CAMELYON16 protocol: mean sensitivity at 1/4, 1/2, 1, 2, 4, 8 FP/slide. Detection points from the official-style procedure on a patch-score heatmap, with discarded patches scoring 0; the point-extraction rule is stated. Only for arms that score every patch | Comparability with prior work | CAMELYON16 |

FROC notes: the official `Evaluation_FROC.py` (computationalpathologygroup/Camelyon16, checked 2026-11-01) builds its
lesions on a level-5 raster (0.243 µm/px × 32), dilates the tumour mask by 37.5 µm (so parts < 75 µm apart fuse), and
calls a region ITC if its fitted-ellipse **major axis on the dilated region** is < 275 µm (= 200 µm + 75 µm of
dilation). ITC regions are ignored (neither TP nor FP), so ITC results never appear in FROC. Use exactly this rule for
FROC and nothing else. Neither the code nor the evaluation page mentions Test_049 or Test_114. test_049 was removed
from the release by the organisers (duplicate); test_114 is kept (no official reason to drop it; see deviations.md).

**Two size systems.** Lesion size classes everywhere except FROC use the clinical (AJCC) rule: greatest dimension (max
Feret) of the undilated lesion, ITC ≤ 0.2 mm, micro > 0.2–2.0 mm, macro > 2.0 mm. CAMELYON17's evaluation page states
the same cutoffs (ITC "smaller than 0.2 mm", micro "greater than 0.2 mm … smaller than 2.0 mm") but leaves exactly
0.2 mm undefined; AJCC breaks that tie. Report lesion counts under both systems (clinical and FROC-code) once, so
readers can see how many lesions change class.

---

## Q3 — Where was it lost? (Tier 3)

**Probe (evaluation instrument only).** Per CV fold, a logistic-regression probe on the **frozen encoder's input
features** (UNI or CONCH), trained on training-fold patch labels. Everything after the frozen encoder, including the
in-encoder pruner's aggregator layers, counts as the selector stage.

| Metric | Definition | Role |
|--------|-----------|------|
| Probe AUC_ℓ | AUC of probe scores for H_ℓ vs the same slide's normal patches. With n_ℓ = 1 this is a percentile rank; report it but flag it | Did the encoder separate this lesion? |
| Probe ESI_ℓ | (wᵀμ_t − wᵀμ_n)² / ½(Var(wᵀx_t) + Var(wᵀx_n)) along probe direction w. **Only for n_ℓ ≥ 2** | Separability comparable between UNI (1024-d) and CONCH (512-d) |
| Layer-of-Death | per lesion, the layer at which its last tumour patch is dropped. Shown as a **Kaplan–Meier curve** of lesion survival across layers (lesions never dropped are censored at the last layer) | In-encoder pruner only |

**Stage attribution, two levels.**

*Lesion level* — every micro lesion with n_ℓ ≥ 1, at K = 32:

| Outcome | Rule |
|---------|------|
| Encoder-stage loss | probe AUC_ℓ < 0.90 |
| Selector-stage loss | probe AUC_ℓ ≥ 0.90 and not retained (H_ℓ ∩ S = ∅) |
| Retained | H_ℓ ∩ S ≠ ∅ |

Micro lesions with n_ℓ = 0 are reported as "unhittable" and are not attributed. Sensitivity analysis at 0.85 and
0.95; report the four counts (encoder, selector, retained, unhittable) with denominators.

*Slide level* — only slides whose largest lesion is micro (so slide detection can be credited to micro lesions):
classifier-stage loss = at least one micro lesion retained **and** slide missed (Protocol C threshold).

**H2 test.** Null: selector share ≤ encoder share among micro lesions that were not retained. Estimate
(selector share − encoder share) with a cluster bootstrap over slides; H2 holds if the 95% lower bound > 0.

---

## Q4 — Did the model use what it kept? (Tier 4, Protocol C)

| Metric | Definition | Role |
|--------|-----------|------|
| **Deletion drop (one lesion)** | Δ slide score when the retained tumour patches of lesion ℓ are removed from S and replaced by the arm's next-ranked non-tumour patches, so \|S\| stays K | Catches "kept but not used" |
| **Deletion drop (all lesions)** | the same, removing every retained tumour patch of the slide | On multi-lesion slides one lesion may be redundant |
| Deletion control | the same number of random retained normal patches removed | So the drop is not just a budget effect |

Not defined for Random (no ranking); for Random, replacements are drawn uniformly from the unretained normal patches.
For merging arms, deletion means re-running the merge without the lesion's patches, which changes W; say so.
Replaces LRP faithfulness. Source: deletion/insertion curves (RISE, Petsiuk et al. 2018) *(verify)*.

---

## Statistics

- **Estimand.** Lesion-level metrics are lesion-pooled within a size class (state this). Per-slide-mean versions are
  sensitivity checks.
- **Uncertainty.** Hierarchical bootstrap: resample slides, then training seeds within each slide's fold (B = 2000,
  seeded). This keeps seed variance in the CIs.
- **Paired comparisons between arms** (H3), on the same pooled estimand:
  - CI: paired cluster bootstrap of the difference in lesion-pooled micro LSR@32.
  - Test: slide-level sign-flip permutation test on per-slide differences in micro-lesion hit counts (the sum of these
    differences, divided by the number of micro lesions, is exactly the pooled difference). Slides without micro
    lesions contribute nothing and are not "ties".
  - Holm across the pre-specified family. A GEE/mixed logistic model of lesion outcomes clustered by slide is the
    secondary analysis.
- **Count behind every rate** (lesions, slides, unhittable lesions).
- ITC: descriptive only (counts and per-lesion outcomes), no tests; ITCs do enter the D50 fit.
- **Power check before any run:** from the actual numbers of OOF micro lesions and slides, compute the expected CI
  width for micro LSR@32 under plausible LSR values. If too wide, enlarge the development set before freezing.

## Primary vs secondary

| Level | Metrics |
|-------|---------|
| Primary (confirmatory) | Micro LSR@32 vs Oracle-LSR-micro (H1); selector − encoder share at K = 32 (H2); paired micro LSR@32 vs Top-K ABMIL (H3) |
| Secondary (CIs, no p-values) | D50, retention AUC, NRS, Cov, TSS, stratum sensitivity, localised sensitivity, deletion drops, Layer-of-Death curves, all other K |
| Diagnostic | TEP, K/N |
| Appendix | overlap > 0 and eroded/dilated hit rules, per-slide-mean LSR, W > τ_W retention rule, SDI, full-dimension ESI / D, FROC |

## Source corrections (from the review)

- MedTokenBudget's "lesion retention rate" is the fraction of lesion **patches** kept: it supports Cov@K, not LSR@K.
- LROC: Starr/Metz and Swensson. Chakraborty: FROC, AFROC, JAFROC.
- Still to verify against primary sources: BraTS 2023 lesion-wise metrics, LUNA16 CPM, Metrics Reloaded, RISE,
  the CAMELYON16 `evaluate.py` ITC cutoff and Test_049 / Test_114 handling.

## Changes from CLAUDE.md (to log in deviations.md once agreed)

- Added: D50, first-hit rank, retention AUC, NRS, TSS, localised sensitivity, deletion tests, Oracle-LSR and
  Oracle-LSR-micro, the identity-free kept-token probe, Kaplan–Meier Layer-of-Death.
- Replaced: SDI → TSS (SDI to appendix); full-dimension ESI + per-slide SVM → cross-slide probe AUC and probe ESI;
  LRP faithfulness → deletion tests; Wilcoxon → sign-flip permutation + paired cluster bootstrap; Random seeds → exact
  expectation.
- Changed: size-class boundaries to AJCC (ITC ≤ 0.2 mm; done in code and deviations.md on 2026-11-01); bootstrap B = 1000 → 2000,
  hierarchical over slides and seeds; non-exhaustively annotated slides excluded from lesion-level metrics.
- Unchanged: the 10% overlap rule for patch labels and lesion hits (CLAUDE.md already uses it).
