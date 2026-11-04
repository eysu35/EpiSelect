#!/bin/bash
# Create a local virtualenv and install EpiSelect with its dependencies.
#   bash requirements.sh        # dev install, CPU jax (pulled in by equinox/optax)
#   bash requirements.sh tpu    # dev install + jax[tpu]
# On a TPU VM use scripts/tpu_setup.sh instead — it provisions the host as well.
set -euo pipefail

cd "$(dirname "$0")"

EXTRAS=dev
if [[ "${1:-}" == "tpu" ]]; then
  EXTRAS=dev,tpu
fi

if command -v uv >/dev/null 2>&1; then
  uv venv .venv                                    # picks the version in .python-version
  uv pip install --python .venv/bin/python -e ".[$EXTRAS]"
else
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -e ".[$EXTRAS]"
fi

echo
echo "Done. Activate with: source .venv/bin/activate"
echo "W&B: export WANDB_API_KEY=... (or write the key to .wandbkey)"
