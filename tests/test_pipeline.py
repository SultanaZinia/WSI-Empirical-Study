import json

import h5py
import numpy as np
import yaml

from lesionbench.config import ROOT, load_config
from lesionbench.preprocess.pipeline import preprocess_slide, write_patch_h5

CFG = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())


def _slide(tmp_path, with_tumor=True):
    # Level-0 slide 4000 x 2000 px at 0.25 um/px -> 512 px patches at 0.5 um/px.
    # Thumbnail at downsample 100 (40 x 20); tissue covers level-0 x 1000-3000, y 0-2000.
    thumb = np.full((20, 40, 3), 235, dtype=np.uint8)
    thumb[:, 10:30] = (220, 140, 190)
    xml = None
    if with_tumor:
        xml = tmp_path / "s.xml"
        xml.write_text(
            '<ASAP_Annotations><Annotations><Annotation Name="a" Type="Polygon" PartOfGroup="Tumor">'
            "<Coordinates>" + "".join(
                f'<Coordinate Order="{i}" X="{x}" Y="{y}" />'
                for i, (x, y) in enumerate([(1100, 100), (1500, 100), (1500, 500), (1100, 500)])
            ) + "</Coordinates></Annotation></Annotations></ASAP_Annotations>"
        )
    return preprocess_slide("s", thumb, 100.0, (4000, 2000), 0.25, xml, CFG)


def test_preprocess_slide_grid_and_labels(tmp_path):
    res = _slide(tmp_path)
    a = res["attrs"]
    assert a["patch_size_level0"] == 512
    # Columns 2..5 (x 1024..2560) and 0 (x 512..1024 touches tissue at 1000) -> cells x in {512..2560};
    # rows y in {0, 512, 1024} (1536 + 512 > 2000 is dropped).
    xs = sorted(set(res["coords"][:, 0].tolist()))
    assert xs == [512, 1024, 1536, 2048, 2560] and sorted(set(res["coords"][:, 1].tolist())) == [0, 512, 1024]
    # Lesion 1100-1500 x 100-500 (400 x 400) sits inside patch (1024, 0): overlap 160000 / 262144 = 0.61.
    pos = res["coords"][res["patch_labels"] == 1].tolist()
    assert pos == [[1024, 0]]
    assert set(res["lesion_ids"].tolist()) == {-1, 0}
    assert a["n_lesions"] == 1 and a["stratum"] == "itc"  # diag 566 px * 0.25 um = 0.141 mm
    assert a["magnification"] == 20.0
    assert json.loads(a["preprocess_config"])["lesion_merge_um"] == 75


def test_preprocess_slide_negative(tmp_path):
    res = _slide(tmp_path, with_tumor=False)
    assert res["patch_labels"].sum() == 0 and (res["lesion_ids"] == -1).all()
    assert res["attrs"]["stratum"] == "negative" and len(res["coords"]) == 15


def test_write_patch_h5_schema_and_atomic(tmp_path):
    res = _slide(tmp_path)
    out = tmp_path / "patches" / "s.h5"
    write_patch_h5(out, res, git_sha="abc")
    assert not list(out.parent.glob("*.tmp"))
    with h5py.File(out) as f:
        assert f["coords"].dtype == np.int64 and f["coords"].shape == (len(res["coords"]), 2)
        assert f["patch_labels"].dtype == np.uint8 and f["lesion_ids"].dtype == np.int32
        np.testing.assert_array_equal(f["coords"][:], res["coords"])
        assert f.attrs["slide_id"] == "s" and f.attrs["git_sha"] == "abc" and f.attrs["stratum"] == "itc"


def test_load_config_env_overrides(tmp_path, monkeypatch):
    server = tmp_path / "server.yaml"
    paths = {"data_root": "/a", "cache_root": "/b"}
    server.write_text(yaml.safe_dump({"slurm": {"account": "def-x"}, "paths": paths}))
    monkeypatch.setenv("CACHE_ROOT", "/env/cache")
    cfg = load_config(server)
    assert cfg["paths"] == {"data_root": "/a", "cache_root": "/env/cache"}
    assert cfg["budgets"] == [32, 64, 128, 256]
