#!/bin/bash
# Frozen-encoder features for every slide. Weights must already be in HF_HOME (scripts/download_weights.py).
# Re-submit to resume: finished slides are skipped.
# Submit from the repo root:
#   sbatch --account=$SLURM_ACCOUNT slurm/extract_features.sbatch configs/server.yaml uni
#   VENV=$HOME/envs/lesionbench-conch sbatch --account=$SLURM_ACCOUNT slurm/extract_features.sbatch configs/server.yaml conch
#SBATCH --job-name=features
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-15
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

CONFIG=${1:-configs/server.yaml}
BACKBONE=${2:-uni}

module load StdEnv/2023 python/3.11 openslide/4.0.0
source "${VENV:-$HOME/envs/lesionbench}/bin/activate"

# Compute nodes have no internet: everything must be pre-downloaded on a login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID host=$(hostname) sha=$(git rev-parse --short HEAD) backbone=$BACKBONE"

python scripts/extract_features.py --config "$CONFIG" --backbone "$BACKBONE" \
    --task-id "$SLURM_ARRAY_TASK_ID" --n-tasks "$SLURM_ARRAY_TASK_COUNT" --num-workers 14
