#!/bin/bash
# External validation: one method's saved CAMELYON16 fold models on another dataset (no training), the 5 folds
# as array tasks. Re-submit to resume. Submit from the repo root:
#   sbatch --account=$SLURM_ACCOUNT slurm/run_external.sbatch configs/c17.local.yaml configs/server.yaml uni lts
# SLICEChat needs its own environment:
#   VENV=$HOME/envs/lesionbench-mamba sbatch ... slurm/run_external.sbatch ... slicechat
#SBATCH --job-name=external
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

CONFIG=${1:?target config}
SOURCE=${2:?source config}
BACKBONE=${3:-uni}
METHOD=${4:?method}
shift 4 || true

module load StdEnv/2023 python/3.11 openslide/4.0.0
source "${VENV:-$HOME/envs/lesionbench}/bin/activate"
export HF_HUB_OFFLINE=1

echo "job=$SLURM_JOB_ID fold=$SLURM_ARRAY_TASK_ID host=$(hostname) sha=$(git rev-parse --short HEAD) method=$METHOD"
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'no usable GPU on $(hostname)')"

python scripts/run_external.py --config "$CONFIG" --source-config "$SOURCE" --backbone "$BACKBONE" \
    --method "$METHOD" --fold "$SLURM_ARRAY_TASK_ID" "$@"
