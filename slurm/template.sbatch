#!/bin/bash
# Template SLURM job for Fir. Copy and adapt per stage; keep logic in scripts/, not here.
# Submit:  sbatch --account=$SLURM_ACCOUNT slurm/template.sbatch configs/server.yaml
#SBATCH --job-name=lesionbench
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=logs/%x-%A_%a.out
# For per-slide work use an array, e.g.  #SBATCH --array=0-398%50

set -euo pipefail

CONFIG=${1:-configs/server.yaml}

module load StdEnv/2023 python/3.11
source "$HOME/envs/lesionbench/bin/activate"

# Compute nodes have no internet: everything must be pre-downloaded on a login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Stage inputs to fast node-local disk if needed:
#   cp "$CACHE_ROOT"/<files> "$SLURM_TMPDIR"/

echo "job=$SLURM_JOB_ID array=${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) sha=$(git rev-parse --short HEAD)"

# python scripts/<stage>.py --config "$CONFIG" --task-id "${SLURM_ARRAY_TASK_ID:-0}"
