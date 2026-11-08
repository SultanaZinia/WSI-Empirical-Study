#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%j.out
# Usage (from the repo root): sbatch --account=$SLURM_ACCOUNT --job-name=<name> midl/slurm/cpu_script.sbatch <script.py> [args...]
module load StdEnv/2023 python/3.11
source ~/envs/lesionbench/bin/activate
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python "$@"
