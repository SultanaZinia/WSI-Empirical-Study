"""Download encoder weights into HF_HOME. Run on a Fir login node (compute nodes have no internet).

    python scripts/download_weights.py --config configs/server.yaml uni conch

Needs a Hugging Face token with access to the gated repos: run `hf auth login` once with the same HF_HOME.
"""

import argparse
import os

from lesionbench.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("backbones", nargs="+", choices=["uni", "conch"])
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", load_config(args.config)["paths"]["hf_home"])  # before importing HF

    from huggingface_hub import hf_hub_download

    from lesionbench.features.encoders import HF_REPOS

    for name in args.backbones:
        print(name, hf_hub_download(*HF_REPOS[name]))


if __name__ == "__main__":
    main()
