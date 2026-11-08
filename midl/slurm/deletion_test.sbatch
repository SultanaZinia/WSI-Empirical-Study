#!/bin/bash
# Lesion-deletion test: one method's saved CAMELYON16 fold models (no training), the 5 folds as array tasks.
# Re-submit to resume (finished folds are skipped). Submit from the repo root:
#   sbatch --account=$SLURM_ACCOUNT midl/slurm/deletion_test.sbatch configs/server.yaml uni lts $OUT
# SLICEChat needs its own environment: VENV=$HOME/envs/lesionbench-mamba sbatch ... slicechat $OUT
#SBATCH --job-name=deletion
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

CONFIG=${1:?config}
BACKBONE=${2:-uni}
METHOD=${3:?method}
OUT=${4:?output dir}

module load StdEnv/2023 python/3.11 openslide/4.0.0
source "${VENV:-$HOME/envs/lesionbench}/bin/activate"
export HF_HUB_OFFLINE=1

echo "job=$SLURM_JOB_ID fold=$SLURM_ARRAY_TASK_ID host=$(hostname) sha=$(git rev-parse --short HEAD) method=$METHOD"
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'no usable GPU on $(hostname)')"

python midl/scripts/deletion_test.py run --config "$CONFIG" --backbone "$BACKBONE" --method "$METHOD" \
    --fold "$SLURM_ARRAY_TASK_ID" --out "$OUT"
