#!/bin/bash
# Methods checks for the paper (2026-11-16), CPU only, reads saved runs, writes to dated folders:
#   1. kept-patch audit of every saved run           -> OUTPUT_ROOT/kept_audit_<date>/
#   2. FROC port vs official evaluate.py, 4 runs     -> OUTPUT_ROOT/froc_official_check_<date>/
#   3. FROC rerun, all methods (incl. DTFD, HistoSelect) -> OUTPUT_ROOT/froc_<date>/uni/
# Step 2 needs pandas in ~/envs/lesionbench (pip install --no-index pandas; nothing else changes).
# Submit from the repo root:
#   sbatch --account=$SLURM_ACCOUNT slurm/paper_checks.sbatch configs/server.yaml 2026-11-16
#SBATCH --job-name=paper_checks
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

CONFIG=${1:-configs/server.yaml}
DATE=${2:?run date, e.g. 2026-11-16}

module load StdEnv/2023 python/3.11 openslide/4.0.0
source "$HOME/envs/lesionbench/bin/activate"
OUT=$(python -c "from lesionbench.config import load_config; print(load_config('$CONFIG')['paths']['output_root'])")

echo "job=$SLURM_JOB_ID host=$(hostname) sha=$(git rev-parse --short HEAD) out=$OUT date=$DATE"
for d in "kept_audit_$DATE" "froc_official_check_$DATE" "froc_$DATE"; do
    [ -e "$OUT/$d" ] && { echo "$OUT/$d exists; not overwriting"; exit 1; }
done

python scripts/audit_kept.py --config "$CONFIG" --out "$OUT/kept_audit_$DATE"
python scripts/froc_official_check.py --config "$CONFIG" --official-python "$(command -v python)" \
    --out "$OUT/froc_official_check_$DATE"
python scripts/run_froc.py --config "$CONFIG" --backbone uni --out-name "froc_$DATE"
