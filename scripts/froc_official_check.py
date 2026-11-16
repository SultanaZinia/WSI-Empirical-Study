"""Cross-check our FROC port (metrics/froc.py) against the official CAMELYON16 evaluate.py.

    python scripts/froc_official_check.py --config configs/server.yaml \
        --official-python ENV/bin/python --out DIR

For a few saved runs, the kept patches become detections exactly as in run_froc.py (full-bag ABMIL scorer,
local maxima). The same detections are scored twice:
- official: DATA_ROOT/evaluation/evaluate.py in a subprocess, with one keyword removed for scikit-image 0.26
  (label(input=...) -> label(...); written to DIR/evaluate_compat.py and evaluate_compat.txt). ASAP
  (multiresolutionimageinterface) is not available on Fir, so a stand-in module provides it: mask pixels via
  tifffile (OpenSlide cannot decode the masks), level downsample and pixel spacing via OpenSlide metadata
  (our port takes them from tifffile).
- ours: metrics/froc.py, as in run_froc.py.
Prints both scores per run, with and without ITCs, and writes DIR/official_check.csv. Reads only.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_froc import test_set  # noqa: E402

from lesionbench.config import load_config  # noqa: E402
from lesionbench.eval.froc import load_scorer  # noqa: E402
from lesionbench.metrics.froc import (  # noqa: E402
    detections_from_kept,
    froc_curve,
    froc_score,
    slide_probabilities,
    to_level,
)

RUNS = [("abmil_full", 0, "full"), ("siimil", 0, "32"), ("tc_ssa", 0, "256"), ("slicechat", 0, "32")]

SHIM = '''"""Stand-in for ASAP's multiresolutionimageinterface (only what evaluate.py calls)."""
import numpy as np
import openslide
import tifffile


class _Image:
    def __init__(self, path):
        self.path, self.slide = path, openslide.OpenSlide(path)

    def getLevelDownsample(self, level):
        return self.slide.level_downsamples[level]

    def getLevelDimensions(self, level):
        return self.slide.level_dimensions[level]

    def getSpacing(self):
        p = self.slide.properties
        return [float(p["openslide.mpp-x"]), float(p["openslide.mpp-y"])]

    def getUCharPatch(self, startX, startY, width, height, level):
        with tifffile.TiffFile(self.path) as t:
            a = t.series[0].levels[level].asarray()
        assert (startX, startY) == (0, 0) and a.shape[:2] == (height, width), (a.shape, width, height)
        return a.reshape(height, width, -1).astype(np.uint8)

    def close(self):
        self.slide.close()


class MultiResolutionImageReader:
    def open(self, path):
        return _Image(path)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--official-python", required=True,
                    help="python with pandas, matplotlib, scikit-image, openslide")
    ap.add_argument("--backbone", default="uni")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    data, out = Path(cfg["paths"]["data_root"]), Path(args.out)
    run_root = Path(cfg["paths"]["output_root"]) / "runs" / args.backbone
    (out / "shim").mkdir(parents=True, exist_ok=True)
    (out / "shim" / "multiresolutionimageinterface.py").write_text(SHIM)
    # scikit-image >= 0.26 no longer accepts label(input=...); the only change to the official code
    src = (data / "evaluation" / "evaluate.py").read_text()
    old, new = "skimage.measure.label(input=", "skimage.measure.label("
    assert src.count(old) == 1, "official evaluate.py differs from the release checked on 2026-11-16"
    evaluate = out / "evaluate_compat.py"
    evaluate.write_text(src.replace(old, new))
    (out / "evaluate_compat.txt").write_text(f"evaluate_compat.py = official evaluate.py with one change:\n"
                                             f"  {old}  ->  {new}\n(keyword removed in scikit-image 0.26)\n")

    slides, masks = test_set(cfg, args.backbone)
    rows = []
    for method, fold, K in RUNS:
        score = load_scorer("abmil", run_root, fold, next(iter(slides.values()))[0].shape[1],
                            cfg["training"]["dropout"])
        kept = np.load(run_root / method / f"fold{fold}" / f"K{K}" / "retained.npz")
        dets = {sid: detections_from_kept(kept[f"retained/{sid}"], c, ps, score(x))
                for sid, (x, c, ps) in slides.items()}
        det_dir = out / "detections" / f"{method}_fold{fold}_K{K}"
        det_dir.mkdir(parents=True, exist_ok=True)
        for sid, d in dets.items():
            with open(det_dir / f"{sid}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["p", "x", "y"])
                w.writerows(d)
        for itc in (False, True):
            per_slide = []
            for sid, d in dets.items():
                if sid in masks:
                    em, itc_labels, ds = masks[sid][itc]
                    per_slide.append(slide_probabilities(to_level(d, ds), em, itc_labels))
                else:
                    per_slide.append(slide_probabilities(d, None, set()))
            ours = froc_score(*froc_curve(per_slide))
            cmd = [args.official_python, str(evaluate),
                   "-r", str(data / "evaluation" / "reference.csv"), "-m", str(data / "masks"),
                   "-d", str(det_dir)] + (["-t"] if itc else [])
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 env={**os.environ, "PYTHONPATH": str(out / "shim"), "MPLBACKEND": "Agg"})
            m = re.search(r"Score: ([0-9.eE+-]+)", res.stdout)
            if not m:
                print(res.stdout[-2000:], res.stderr[-4000:], sep="\n")
                raise SystemExit(f"official script failed for {method} K{K} itc={itc}")
            official = float(m.group(1))
            rows.append({"method": method, "fold": fold, "K": K, "itc": int(itc), "ours": ours,
                         "official": official, "abs_diff": abs(ours - official)})
            print(rows[-1], flush=True)
    with open(out / "official_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("max |ours - official|:", max(r["abs_diff"] for r in rows))


if __name__ == "__main__":
    main()
