"""OpenSlide reads. Kept apart from the rest of preprocess so tests never need OpenSlide."""

import numpy as np
import openslide


def read_thumbnail(slide: openslide.OpenSlide, thumb_mpp: float) -> tuple[np.ndarray, float, float]:
    """Whole slide at the pyramid level nearest ``thumb_mpp`` -> (RGB, level downsample, level-0 mpp)."""
    mpp0 = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    level = slide.get_best_level_for_downsample(thumb_mpp / mpp0)
    img = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
    return np.asarray(img), float(slide.level_downsamples[level]), mpp0
