#!/bin/bash
#SBATCH --job-name=build_cp_tfds
#SBATCH --account=<your_account>
#SBATCH --output=/path/to/EpiSelect_out/build_cp_tfds_%A_%a.out
#SBATCH --error=/path/to/EpiSelect_out/build_cp_tfds_%A_%a.err
#SBATCH --array=0-30
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# One array task per domain. Builds to local scratch, then rsyncs that domain to the bucket.
# Submit a subset with e.g.  sbatch --array=3,7 scripts/build_commonpile_tfds.slurm
set -euo pipefail

REPO=${REPO:-/scratch/$USER/EpiSelect}
DATA_DIR=${DATA_DIR:-/scratch/$USER/commonpile_tfds_v3}
BUCKET=${BUCKET:-gs://commonpile-tfds/commonpile_tfds}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export TOKENIZERS_PARALLELISM=true

cd "$REPO"
DOMAIN=$(.venv/bin/python scripts/build_commonpile_tfds.py --list | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "task $SLURM_ARRAY_TASK_ID -> $DOMAIN on $(hostname)"

.venv/bin/python scripts/build_commonpile_tfds.py --domains "$DOMAIN" --data_dir "$DATA_DIR" --sync "$BUCKET"
