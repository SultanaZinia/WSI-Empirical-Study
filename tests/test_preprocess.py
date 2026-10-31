import logging

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from lesionbench.preprocess.annotations import lesions, max_feret, read_asap_xml, stratum
from lesionbench.preprocess.grid import patch_grid, patch_size_level0
from lesionbench.preprocess.labels import label_patches
from lesionbench.preprocess.tissue import otsu_threshold, tissue_mask

# --- tissue ---------------------------------------------------------------------------------------


def test_otsu_splits_two_modes():
    values = np.array([50] * 100 + [200] * 100, dtype=np.uint8)
    t = otsu_threshold(values)
    assert 50 <= t < 200
    assert ((values > t) == (values == 200)).all()


def test_otsu_constant_input_has_empty_upper_class():
    values = np.full(10, 7, dtype=np.uint8)
    assert not (values > otsu_threshold(values)).any()


def test_tissue_mask_drops_glass_and_adipose():
    thumb = np.full((20, 20, 3), 235, dtype=np.uint8)  # glass
    thumb[5:15, 5:15] = (220, 140, 190)  # H&E-pink tissue
    thumb[8:10, 8:10] = (250, 250, 250)  # adipose hole inside tissue
    expected = np.zeros((20, 20), dtype=bool)
    expected[5:15, 5:15] = True
    expected[8:10, 8:10] = False
    np.testing.assert_array_equal(tissue_mask(thumb), expected)


def test_tissue_mask_blank_slide_is_empty():
    assert not tissue_mask(np.full((8, 8, 3), 240, dtype=np.uint8)).any()


# --- grid -----------------------------------------------------------------------------------------


def test_patch_size_level0_camelyon16_scanners():
    assert patch_size_level0(256, 0.5, 0.243) == 527  # 526.75
    assert patch_size_level0(256, 0.5, 0.226) == 566  # 566.37


def test_patch_grid_keeps_cells_touching_tissue():
    # Slide 100 x 50 level-0 px, 20 px patches -> 5 x 2 full cells (the 10 px bottom strip is dropped).
    # Mask at downsample 10 -> shape (5, 10); one tissue pixel covers level-0 x 30-40, y 0-10,
    # which is 1 of the 4 mask pixels of cell (20, 0) -> tissue fraction 0.25.
    mask = np.zeros((5, 10), dtype=bool)
    mask[0, 3] = True
    mask[4, 0] = True  # in the partial bottom strip: no full patch there
    kw = dict(mask=mask, mask_downsample=10, level0_size=(100, 50), patch_size_l0=20)
    np.testing.assert_array_equal(patch_grid(**kw), [[20, 0]])
    np.testing.assert_array_equal(patch_grid(**kw, min_tissue_fraction=0.25), [[20, 0]])
    assert patch_grid(**kw, min_tissue_fraction=0.3).shape == (0, 2)


def test_patch_grid_row_major_order_and_dtype():
    coords = patch_grid(np.ones((5, 10), dtype=bool), 10, (100, 50), 20)
    assert coords.dtype == np.int64
    assert coords[:6].tolist() == [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [0, 20]]
    assert len(coords) == 10


# --- annotations ----------------------------------------------------------------------------------


def _asap(annotations):
    body = []
    for name, typ, group, pts in annotations:
        # write coordinates in reverse Order to check that the parser sorts them
        coords = "".join(
            f'<Coordinate Order="{i}" X="{x}" Y="{y}" />' for i, (x, y) in reversed(list(enumerate(pts)))
        )
        body.append(f'<Annotation Name="{name}" Type="{typ}" PartOfGroup="{group}">'
                    f"<Coordinates>{coords}</Coordinates></Annotation>")
    return ('<?xml version="1.0"?><ASAP_Annotations><Annotations>' + "".join(body)
            + '</Annotations><AnnotationGroups><Group Name="Tumor" PartOfGroup="None" />'
            "</AnnotationGroups></ASAP_Annotations>")


SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_read_asap_xml_groups_and_skips(tmp_path, caplog):
    xml = _asap([
        ("t0", "Polygon", "Tumor", SQUARE),
        ("t1", "Polygon", "_1", [(500, 500), (510, 500), (510, 510)]),
        ("hole", "Polygon", "Exclusion", [(40, 40), (60, 40), (60, 60), (40, 60)]),
        ("dot", "Dot", "None", [(5, 5)]),  # not a lesion group: ignored silently
        ("tdot", "Dot", "Tumor", [(5, 5)]),  # tumor dot: skipped with a warning
    ])
    p = tmp_path / "s.xml"
    p.write_text(xml)
    with caplog.at_level(logging.WARNING):
        tumor, exclusion = read_asap_xml(p)
    assert [t.area for t in tumor] == [10000, 50]
    assert [e.area for e in exclusion] == [400]
    assert "tdot" in caplog.text and "'dot'" not in caplog.text


def test_read_asap_xml_repairs_bowtie(tmp_path):
    p = tmp_path / "s.xml"
    p.write_text(_asap([("bow", "Polygon", "Tumor", [(0, 0), (10, 10), (10, 0), (0, 10)])]))
    (poly,), _ = read_asap_xml(p)
    assert poly.is_valid


def test_lesions_union_exclusion_and_order():
    tumor = [box(0, 1000, 100, 1100),  # lower on the slide -> id 1
             box(0, 0, 100, 100), box(50, 0, 150, 100)]  # overlapping -> one lesion, id 0
    comps = lesions(tumor, [box(40, 40, 60, 60)])
    assert len(comps) == 2
    assert comps[0].area == 150 * 100 - 400
    assert comps[1].bounds == (0, 1000, 100, 1100)


def test_lesions_exclusion_splits_a_lesion():
    comps = lesions([box(0, 0, 300, 100)], [box(100, 0, 200, 100)])
    assert [c.bounds for c in comps] == [(0, 0, 100, 100), (200, 0, 300, 100)]


def test_lesions_negative_slide():
    assert lesions([], []) == []
    assert lesions([], [], merge_px=10) == []


def test_lesions_merge_nearby_regions():
    # Gaps: a-b 10 px, b-c 10 px (chain), c-d 30 px.
    a, b, c, d = box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10), box(80, 0, 90, 10)
    assert len(lesions([a, b, c, d], [])) == 4
    assert len(lesions([a, b, c, d], [], merge_px=9.9)) == 4
    merged = lesions([d, c, b, a], [], merge_px=10)  # a-b-c chain into one lesion; d stays apart
    assert len(merged) == 2
    assert merged[0].area == 300 and merged[0].geom_type == "MultiPolygon"
    assert max_feret(merged[0]) == pytest.approx((50**2 + 10**2) ** 0.5)  # spans the whole group
    assert merged[1].bounds == (80, 0, 90, 10)


def test_merge_changes_stratum():
    # At 0.25 um/px (1 mm = 4000 px): two 400 px squares (diag 0.141 mm, ITC) 200 px = 0.05 mm apart.
    # Merged, the group spans 1000 x 400 px -> 0.269 mm, micro.
    a, b = box(0, 0, 400, 400), box(600, 0, 1000, 400)
    assert stratum(lesions([a, b], []), mpp0=0.25) == "itc"
    assert stratum(lesions([a, b], [], merge_px=75 / 0.25), mpp0=0.25) == "micro"


def test_max_feret_rectangle_diagonal():
    assert max_feret(box(0, 0, 3, 4)) == pytest.approx(5.0)
    assert max_feret(Polygon([(0, 0), (10, 0), (5, 1)])) == pytest.approx(10.0)


@pytest.mark.parametrize("w,h,expected", [
    (100, 100, "itc"),      # diag 141 px = 0.035 mm
    (480, 640, "micro"),    # diag 800 px = 0.2 mm exactly -> micro
    (4800, 6400, "micro"),  # diag 8000 px = 2.0 mm exactly -> micro
    (6000, 6400, "macro"),
])
def test_stratum_thresholds(w, h, expected):
    assert stratum([box(0, 0, w, h)], mpp0=0.25) == expected  # 1 mm = 4000 px


def test_stratum_uses_largest_lesion_and_negative():
    assert stratum([box(0, 0, 10, 10), box(0, 0, 6000, 6400)], mpp0=0.25) == "macro"
    assert stratum([], mpp0=0.25) == "negative"


# --- labels ---------------------------------------------------------------------------------------

COORDS = np.array([[0, 0], [10, 0], [20, 0]])  # three 10 x 10 patches in a row


def test_label_patches_overlap_threshold():
    # Lesion 0 straddles patches 0 (overlap 20/100) and 1 (30/100).
    # Lesion 1 covers exactly 10/100 of patch 2: at the threshold -> positive.
    labels, ids, unlabeled = label_patches(COORDS, 10, [box(8, 0, 13, 10), box(20, 0, 21, 10)])
    assert labels.tolist() == [1, 1, 1]
    assert ids.tolist() == [0, 0, 1]
    assert labels.dtype == np.uint8 and ids.dtype == np.int32
    assert unlabeled == []


def test_label_patches_largest_overlap_wins_and_itc_is_logged(caplog):
    # Patch 0: lesion 0 overlaps 0.2, lesion 1 overlaps 0.5 -> lesion 1.
    # Lesion 2 is an ITC covering 5/100 of patch 2: below threshold.
    polys = [box(0, 0, 2, 10), box(5, 0, 10, 10), box(21, 0, 22, 5)]
    with caplog.at_level(logging.WARNING):
        labels, ids, unlabeled = label_patches(COORDS, 10, polys)
    assert labels.tolist() == [1, 0, 0]
    assert ids.tolist() == [1, -1, -1]
    assert unlabeled == [0, 2]
    assert "lesion 2" in caplog.text


def test_label_patches_merged_lesion_sums_its_parts():
    # One merged lesion made of two pieces inside patch 1: 6/100 + 6/100 = 0.12 -> positive.
    (lesion,) = lesions([box(10, 0, 13, 2), box(15, 0, 18, 2)], [], merge_px=5)
    labels, ids, unlabeled = label_patches(COORDS, 10, [lesion])
    assert ids.tolist() == [-1, 0, -1] and unlabeled == []


def test_label_patches_negative_slide_and_no_patches():
    labels, ids, unlabeled = label_patches(COORDS, 10, [])
    assert labels.tolist() == [0, 0, 0] and ids.tolist() == [-1, -1, -1] and unlabeled == []
    labels, ids, unlabeled = label_patches(np.zeros((0, 2)), 10, [box(0, 0, 5, 5)])
    assert len(labels) == 0 and unlabeled == [0]
