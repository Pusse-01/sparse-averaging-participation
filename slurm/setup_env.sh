#!/bin/bash
# One-time environment setup. Run on the LOGIN node (needs internet).
#
#     bash slurm/setup_env.sh
#
# Creates a venv at $ASYNCMESH_VENV (default ~/venvs/asyncmesh), installs deps,
# pre-tokenises WikiText-103, and does a CPU smoke test. Compute nodes on this
# cluster have no egress, so everything that touches the network happens here.

set -euo pipefail

VENV="${ASYNCMESH_VENV:-$HOME/venvs/asyncmesh}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> creating venv at $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip

echo "==> installing dependencies"
pip install -r "$REPO/requirements.txt"

echo "==> logging in to Weights & Biases"
if ! wandb status 2>/dev/null | grep -q "Logged in"; then
    echo "    run 'wandb login' now if this is a fresh machine"
    wandb login || echo "    (skipped; set WANDB_API_KEY instead)"
fi

echo "==> preparing data (WikiText-103)"
cd "$REPO"
python scripts/prepare_data.py --dataset wikitext103 --out-dir data

echo "==> verifying the claims the design rests on"
python -m tests.test_claims

echo "==> CPU smoke test"
python -m asyncmesh_sim.train --synthetic --no-wandb --preset tiny --m 2 \
    --p 0.4 --tau 5 --steps 20 --eval-interval 10 --batch-size 2 \
    --vocab-size 512 --device cpu --dtype fp32 --out-dir /tmp/asyncmesh-smoke

cat <<EOF

Setup complete.

Next:
    python -m experiments.grids --all
    mkdir -p logs
    sbatch --array=1-66%12 slurm/run_array.sbatch slurm/manifests/e1.txt

Add this to your shell rc so sbatch picks up the venv:
    export ASYNCMESH_VENV=$VENV
EOF
