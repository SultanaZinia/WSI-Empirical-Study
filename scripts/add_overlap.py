"""Add (patch, lesion, overlap) pairs to existing patch files, from the annotations alone (no slide reads).

    python scripts/add_overlap.py --config configs/server.yaml

For each CACHE_ROOT/patches/<slide>.h5 without a complete /overlap group, rebuilds the slide's lesions from
DATA_ROOT/annotations/<slide>.xml with the merge distance the file was made with, checks the lesion count
against the file, and writes /overlap. Slides without an XML get an empty group. Exits non-zero on failure.
"""

import argparse
import json
import logging
from pathlib import Path

import h5py

from lesionbench.config import load_config
from lesionbench.preprocess.annotations import lesions, read_asap_xml
from lesionbench.preprocess.labels import lesion_overlaps
from lesionbench.preprocess.pipeline import write_overlap

log = logging.getLogger("add_overlap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    ann = Path(cfg["paths"]["data_root"]) / "annotations"
    failed = []
    for p in sorted((Path(cfg["paths"]["cache_root"]) / "patches").glob("*.h5")):
        try:
            with h5py.File(p, "a") as f:
                if f.get("overlap") is not None and f["overlap"].attrs.get("complete", False):
                    continue
                a = dict(f.attrs)
                sid, mpp0 = str(a["slide_id"]), float(a["mpp0"])
                merge_um = json.loads(a["preprocess_config"])["lesion_merge_um"]
                xml = ann / f"{sid}.xml"
                les = lesions(*read_asap_xml(xml), merge_px=merge_um / mpp0) if xml.exists() else []
                if len(les) != int(a["n_lesions"]):
                    raise ValueError(f"{len(les)} lesions rebuilt, file has {a['n_lesions']}")
                pairs = lesion_overlaps(f["coords"][:], int(a["patch_size_level0"]), les)
                write_overlap(f, *pairs)
            log.info("%s: %d lesions, %d overlap pairs", sid, len(les), len(pairs[0]))
        except Exception:
            log.exception("%s: failed", p.stem)
            failed.append(p.stem)
    if failed:
        log.error("failed slides: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
