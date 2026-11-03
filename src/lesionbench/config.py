"""Load configs/experiment.yaml merged with a server config; server paths can be overridden by env vars."""

import json
import os
import subprocess
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


def git_sha() -> str:
    git = ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
    return subprocess.run(git, capture_output=True, text=True).stdout.strip()


def write_run_json(out_dir, cfg: dict, name: str = "run.json", **extra) -> None:
    """Provenance next to an output folder: git SHA, config (incl. seed) and run-specific values."""
    Path(out_dir, name).write_text(json.dumps({"git_sha": git_sha(), **extra, "config": cfg}, indent=1,
                                              default=str))
