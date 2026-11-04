#!/bin/bash
#SBATCH --job-name=count_cp_tokens
#SBATCH --account=<your_account>
#SBATCH --output=/path/to/EpiSelect_out/count_cp_tokens_%A_%a.out
#SBATCH --error=/path/to/EpiSelect_out/count_cp_tokens_%A_%a.err
#SBATCH --array=0-30
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Writes token_counts.json for every already-built domain and syncs it to the bucket. Run after the build
# array; domains that already have the file are skipped.
set -euo pipefail

REPO=${REPO:-/scratch/$USER/EpiSelect}
DATA_DIR=${DATA_DIR:-/scratch/$USER/commonpile_tfds_v3}
BUCKET=${BUCKET:-gs://commonpile-tfds/commonpile_tfds}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

cd "$REPO"
DOMAIN=$(.venv/bin/python scripts/build_commonpile_tfds.py --list | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "task $SLURM_ARRAY_TASK_ID -> $DOMAIN on $(hostname)"

.venv/bin/python scripts/build_commonpile_tfds.py --count_only --domains "$DOMAIN" --data_dir "$DATA_DIR" --sync "$BUCKET"
