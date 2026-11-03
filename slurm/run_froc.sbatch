#!/bin/bash
# FROC under the shared tumour scorers (CPU only). Run after every method's retrain has finished.
# Submit from the repo root:
#   sbatch --account=$SLURM_ACCOUNT slurm/run_froc.sbatch configs/server.yaml uni --scorer-auc
#   sbatch --account=$SLURM_ACCOUNT slurm/run_froc.sbatch configs/server.yaml uni
#SBATCH --job-name=froc
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

CONFIG=${1:-configs/server.yaml}
BACKBONE=${2:-uni}
shift 2 || true

module load StdEnv/2023 python/3.11
source "$HOME/envs/lesionbench/bin/activate"

echo "job=$SLURM_JOB_ID host=$(hostname) sha=$(git rev-parse --short HEAD)"

python scripts/run_froc.py --config "$CONFIG" --backbone "$BACKBONE" "$@"
