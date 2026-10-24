# Activate the project environment. Sourced, not executed.
#
#     source slurm/activate_env.sh
#
# Supports either a conda env or a venv, so the same sbatch scripts work on
# sites that standardise on conda (Monash FITcluster) and sites that do not.
# Precedence:
#
#   ASYNCMESH_CONDA  path to, or name of, a conda environment   -> conda
#   ASYNCMESH_VENV   path to a venv (default ~/venvs/asyncmesh) -> venv
#
# Set exactly one in your shell rc. If ASYNCMESH_CONDA is set it wins.
#
# Neither being set is the common failure: `srun` and `sbatch` export the
# submitting shell's environment, so a variable that lives only in ~/.bashrc
# and was never sourced does not reach the job. slurm/env.conf is the durable
# answer -- it is read from the repo itself, so it survives --export=NONE and
# a fresh login alike. Create it once:
#
#     echo "ASYNCMESH_CONDA=$SCRATCH/envs/asyncmesh" > slurm/env.conf
#
# It is gitignored, since the path is per-user.

_ae_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${ASYNCMESH_CONDA:-}" && -z "${ASYNCMESH_VENV:-}" \
      && -f "$_ae_repo/slurm/env.conf" ]]; then
    # shellcheck disable=SC1091
    set -a; source "$_ae_repo/slurm/env.conf"; set +a
    echo "env config: $_ae_repo/slurm/env.conf"
fi

if [[ -n "${ASYNCMESH_CONDA:-}" ]]; then
    # `conda activate` is a shell function that only exists after the hook is
    # sourced; batch jobs get a non-interactive shell where it is absent.
    if [[ -z "${CONDA_EXE:-}" ]]; then
        for base in "$HOME/miniconda3" "$HOME/anaconda3" \
                    /opt/conda /usr/local/conda; do
            [[ -f "$base/etc/profile.d/conda.sh" ]] && CONDA_EXE="$base/bin/conda" && break
        done
    fi
    if [[ -n "${CONDA_EXE:-}" ]]; then
        # shellcheck disable=SC1091
        source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    fi
    conda activate "$ASYNCMESH_CONDA"
    echo "env       : conda $ASYNCMESH_CONDA ($(which python))"
else
    VENV="${ASYNCMESH_VENV:-$HOME/venvs/asyncmesh}"
    if [[ ! -f "$VENV/bin/activate" ]]; then
        echo "no environment found." >&2
        echo "  set ASYNCMESH_CONDA=<name-or-path> for conda, or" >&2
        echo "  set ASYNCMESH_VENV=<path> for a venv (looked in $VENV)" >&2
        return 1 2>/dev/null || exit 1
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "env       : venv $VENV ($(which python))"
fi
