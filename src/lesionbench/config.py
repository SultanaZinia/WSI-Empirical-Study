"""Load configs/experiment.yaml merged with a server config; server paths can be overridden by env vars."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PATH_ENV = {
    "data_root": "DATA_ROOT",
    "cache_root": "CACHE_ROOT",
    "output_root": "OUTPUT_ROOT",
    "hf_home": "HF_HOME",
}


def load_config(server_path, experiment_path=ROOT / "configs" / "experiment.yaml") -> dict:
    cfg = yaml.safe_load(Path(experiment_path).read_text())
    server = yaml.safe_load(Path(server_path).read_text())
    for key, env in PATH_ENV.items():
        if env in os.environ:
            server["paths"][key] = os.environ[env]
    if "SLURM_ACCOUNT" in os.environ:
        server.setdefault("slurm", {})["account"] = os.environ["SLURM_ACCOUNT"]
    return {**cfg, **server}
