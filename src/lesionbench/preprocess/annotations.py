"""ASAP XML annotations -> tumor lesions as shapely polygons in level-0 pixel coordinates."""

import logging
import xml.etree.ElementTree as ET

from scipy.spatial.distance import pdist
from shapely.geometry import Polygon
from shapely.ops import unary_union

log = logging.getLogger(__name__)

# CAMELYON16 uses "Tumor"/"Exclusion"; "_0"/"_1"/"_2" is the CAMELYON17-style naming.
TUMOR_GROUPS = {"Tumor", "_0", "_1"}
EXCLUSION_GROUPS = {"Exclusion", "_2"}


def read_asap_xml(path) -> tuple[list[Polygon], list[Polygon]]:
    """Return (tumor polygons, exclusion polygons). Invalid polygons are repaired with ``buffer(0)``."""
    tumor, exclusion = [], []
    for ann in ET.parse(path).getroot().iter("Annotation"):
        group = ann.get("PartOfGroup")
        if group not in TUMOR_GROUPS | EXCLUSION_GROUPS:
            continue
        pts = sorted(ann.iter("Coordinate"), key=lambda c: int(c.get("Order")))
        xy = [(float(c.get("X")), float(c.get("Y"))) for c in pts]
        if ann.get("Type") != "Polygon" or len(xy) < 3:
            log.warning("%s: skipping %s annotation %r (type %s, %d points)",
                        path, group, ann.get("Name"), ann.get("Type"), len(xy))
            continue
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        (tumor if group in TUMOR_GROUPS else exclusion).append(poly)
    return tumor, exclusion


def lesions(tumor: list[Polygon], exclusion: list[Polygon]) -> list[Polygon]:
    """Contiguous lesion components (tumor minus exclusions), ordered top-to-bottom then left-to-right.

    The list index is the ``lesion_id``.
    """
    region = unary_union(tumor)
    if exclusion:
        region = region.difference(unary_union(exclusion))
    parts = getattr(region, "geoms", [region])
    comps = [p for p in parts if isinstance(p, Polygon) and p.area > 0]
    return sorted(comps, key=lambda p: (p.bounds[1], p.bounds[0]))


def max_feret(poly: Polygon) -> float:
    """Greatest dimension of a polygon: the largest distance between two of its points (input units)."""
    return float(pdist(list(poly.convex_hull.exterior.coords)).max())


def stratum(lesion_polys: list[Polygon], mpp0: float, itc_max: float = 0.2, micro_max: float = 2.0) -> str:
    """Slide stratum from its largest lesion's max Feret diameter in mm.

    negative: no lesions; itc: < itc_max; micro: itc_max..micro_max; macro: > micro_max.
    """
    if not lesion_polys:
        return "negative"
    mm = max(max_feret(p) for p in lesion_polys) * mpp0 / 1000.0
    if mm > micro_max:
        return "macro"
    return "micro" if mm >= itc_max else "itc"
