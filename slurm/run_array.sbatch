#!/bin/bash
#SBATCH --job-name=asyncmesh
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# stdout and stderr deliberately share one file: with them split, a job that
# died left an .out containing only the preamble while the traceback sat in a
# .err nobody thought to open.
#SBATCH --output=logs/%x_%A_%a.out
#
# One array task = one full mesh run. The m DP replicas are simulated inside a
# single process, so each task needs exactly one GPU -- which is what lets the
# sweep fill both GPUs on every free A100 node at once.
#
# Usage:
#     python -m experiments.grids --all
#     mkdir -p logs
#     sbatch --array=1-66%12 slurm/run_array.sbatch slurm/manifests/e1.txt
#
# The %12 throttle keeps 12 tasks resident, matching the ~6 free nodes x 2
# A100s in the A100 partition. Raise it if more nodes free up; SLURM queues
# the rest.

# No `set -e`: it made every failure silent -- the script exited at the first
# non-zero status without saying which line, and the exit code alone reached
# sacct. Failures are reported explicitly instead.
set -uo pipefail

MANIFEST="${1:?usage: sbatch run_array.sbatch <manifest.txt>}"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
if [[ -z "$LINE" ]]; then
    echo "no line ${SLURM_ARRAY_TASK_ID} in $MANIFEST" >&2
    exit 1
fi

echo "host      : $(hostname)"
echo "gpu       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "manifest  : $MANIFEST line $SLURM_ARRAY_TASK_ID"
echo "args      : $LINE"

# Do NOT `module purge` here. On a Bright-managed site the slurm client lives
# in a module, so purging removes srun/sbatch from PATH and the job dies with
# "srun: command not found" -- after the preamble has already printed, which
# makes it look like a hang in Python.
# Adjust to your site's module names if these do not exist.
module load cuda 2>/dev/null || true

source "${SLURM_SUBMIT_DIR}/slurm/activate_env.sh"
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

# Compute nodes here have no outbound network, so W&B runs offline and is
# synced from the login node afterwards with slurm/sync_wandb.sh.
# Force offline: compute nodes here have no egress, and an online wandb.init
# blocks on retries until it times out, producing a job that burns its
# allocation and writes nothing.
export WANDB_MODE=offline
# Respect an exported WANDB_DIR: offline runs accumulate quickly and sites
# often cap /home, so the submit directory is only the fallback.
export WANDB_DIR="${WANDB_DIR:-${SLURM_SUBMIT_DIR}/wandb}"
# wandb >= 0.18 spawns a core service and waits for it; be explicit about the
# ceiling so a wedged service fails fast instead of hanging the run.
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-60}"
# Unbuffered: without this Python block-buffers stdout to the log file, so a
# job that dies before its first flush leaves an empty log and no traceback.
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-5}"

mkdir -p "$WANDB_DIR" results logs || { echo "FAILED: mkdir under WANDB_DIR=$WANDB_DIR" >&2; exit 1; }

echo "python    : $(command -v python)"
echo "wandb dir : $WANDB_DIR"
echo "launching : python -u -m asyncmesh_sim.train"

# Plain python, not srun. Each array task is a single-task job that already
# holds its GPU allocation, so srun buys nothing here and only adds a
# dependency on the slurm client being on PATH.
python -u -m asyncmesh_sim.train $LINE \
    --out-dir "results/$(basename "$MANIFEST" .txt)"
RC=$?

if [[ $RC -ne 0 ]]; then
    echo "FAILED: training exited $RC" >&2
    exit $RC
fi
echo "done"
