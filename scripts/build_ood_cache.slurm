#!/bin/bash
#SBATCH --job-name=build_ood_cache
#SBATCH --account=<your_account>
#SBATCH --output=/path/to/EpiSelect_out/build_ood_cache_%j.out
#SBATCH --error=/path/to/EpiSelect_out/build_ood_cache_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# 10M-token OOD eval streams (c4, slim_pajama, fine_web), cached as .npy and synced to the bucket.
set -euo pipefail

REPO=${REPO:-/scratch/$USER/EpiSelect}
CACHE_DIR=${CACHE_DIR:-/scratch/$USER/data/ood_token_cache}
BUCKET=${BUCKET:-gs://commonpile-tfds/ood_token_cache}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}

cd "$REPO"
.venv/bin/python scripts/build_ood_cache.py --cache_dir "$CACHE_DIR" --sync "$BUCKET"
