from pathlib import Path

import yaml

import lesionbench

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    assert lesionbench.__version__


def test_experiment_config_parses():
    cfg = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    assert cfg["budgets"] == [32, 64, 128, 256]
    assert cfg["preprocess"]["label_overlap_threshold"] == 0.10


def test_server_example_config_parses():
    cfg = yaml.safe_load((ROOT / "configs" / "server.example.yaml").read_text())
    assert {"data_root", "cache_root", "output_root", "hf_home"} <= cfg["paths"].keys()
