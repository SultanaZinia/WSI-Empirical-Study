#!/bin/bash
# One method, the 5 CV folds as array tasks, every budget K inside each task. Re-submit to resume.
# Submit from the repo root:
#   sbatch --account=$SLURM_ACCOUNT slurm/run_methods.sbatch configs/server.yaml uni lts
# SLICEChat needs its own environment (torch 2.5.1 + mamba_ssm; docs/issues_log.md):
#   VENV=$HOME/envs/lesionbench-mamba sbatch --account=$SLURM_ACCOUNT slurm/run_methods.sbatch configs/server.yaml uni slicechat
#SBATCH --job-name=methods
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

CONFIG=${1:-configs/server.yaml}
BACKBONE=${2:-uni}
METHOD=${3:?method}
shift 3 || true

module load StdEnv/2023 python/3.11 openslide/4.0.0
source "${VENV:-$HOME/envs/lesionbench}/bin/activate"
export HF_HUB_OFFLINE=1

echo "job=$SLURM_JOB_ID fold=$SLURM_ARRAY_TASK_ID host=$(hostname) sha=$(git rev-parse --short HEAD) method=$METHOD"

# Fail fast on a node whose GPU is not usable (fc10512 fell back to CPU on 2026-11-02); resubmit with --exclude.
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'no usable GPU on $(hostname)')"

python scripts/run_methods.py --config "$CONFIG" --backbone "$BACKBONE" --method "$METHOD" \
    --fold "$SLURM_ARRAY_TASK_ID" "$@"
