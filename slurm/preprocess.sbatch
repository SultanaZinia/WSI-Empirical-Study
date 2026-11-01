#!/bin/bash
# Tissue grid + patch labels for all slides (CPU only). Re-submit to resume: finished slides are skipped.
# Submit from the repo root:  sbatch --account=$SLURM_ACCOUNT slurm/preprocess.sbatch configs/server.yaml
#SBATCH --job-name=preprocess
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-15
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

CONFIG=${1:-configs/server.yaml}

module load StdEnv/2023 python/3.11 openslide/4.0.0
source "$HOME/envs/lesionbench/bin/activate"

echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID host=$(hostname) sha=$(git rev-parse --short HEAD)"

python scripts/preprocess.py --config "$CONFIG" --task-id "$SLURM_ARRAY_TASK_ID" --n-tasks "$SLURM_ARRAY_TASK_COUNT"
