# lesionbench

A matched-budget benchmark of token compression for rare lesions in whole-slide images.
It runs on CAMELYON16 and measures whether micro-metastases and isolated tumor cells survive when MIL
selectors, token pruners and slot aggregators compress a slide to K ∈ {32, 64, 128, 256} tokens.

See `CLAUDE.md` for the full experimental spec and conventions.

## Layout

```
src/lesionbench/   preprocess, features, selectors, lineage, metrics, eval, stats, viz
scripts/           config-driven entry points (one per pipeline stage)
slurm/             Fir job wrappers around scripts/
configs/           experiment.yaml (committed), server.example.yaml → server.yaml (git-ignored)
tests/             CPU-only tests on synthetic data; `-m server` tests run on Fir only
docs/              deviations log, notes
```

## Local development (no data, no GPU)

```bash
pip install -e ".[dev]"
pytest
```

## On Fir

```bash
module load StdEnv/2023 python/3.11
virtualenv --no-download $HOME/envs/lesionbench && source $HOME/envs/lesionbench/bin/activate
pip install --no-index -r requirements-fir.txt && pip install --no-index --no-deps -e .
cp configs/server.example.yaml configs/server.yaml   # then edit
mkdir -p logs
sbatch --account=def-<PI> slurm/template.sbatch configs/server.yaml
```
