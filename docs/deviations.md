# Deviations from the planning document

Record every place where the implementation departs from the original plan, including cases where a named
baseline could not be verified and was replaced by a generic family representative.

| Date | Component | Plan said | What we did | Why |
|------|-----------|-----------|-------------|-----|
| 2026-10-31 | Annotations | ASAP `_2` groups are exclusions (holes) | Exclusion groups are `Exclusion` and `_2`; tumor groups are `Tumor`, `_0`, `_1`. Non-polygon annotations are skipped with a warning | The 160 CAMELYON16 XMLs on Fir use only `Tumor` and `Exclusion` (`_0/_1/_2` is CAMELYON17 naming). One `Dot` annotation exists (test_066, group `None`) |
| 2026-10-31 | Patch grid | Not specified: how much tissue a patch needs | Keep a grid cell if its tissue fraction on the thumbnail mask is > 0 (`preprocess.min_tissue_fraction: 0.0`) | Follows the TRIDENT default (`--min_tissue_proportion 0.`); edge patches matter for ITCs and micro-metastases near the capsule |
| 2026-10-31 | Strata | "Largest contiguous lesion diameter" | Max Feret diameter (greatest distance between two points of the lesion), in mm via the slide's level-0 mpp. ITC < 0.2 mm ≤ micro ≤ 2.0 mm < macro | Matches AJCC "greatest dimension"; the plan did not define the diameter |
