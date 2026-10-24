#!/bin/bash
# Get run artefacts off $SCRATCH, which is purge-prone, and record exactly what
# was captured. NEVER deletes anything.
#
#     bash slurm/archive_results.sh
#     ARCHIVE_ROOT=/nfsdata/data/nirodya/archives bash slurm/archive_results.sh
#
# Captures results/, figures/, logs/ and the wandb/ offline runs, plus a manifest
# naming the commit they came from and which W&B runs are still unsynced. Run it
# BEFORE any git checkout, and again after every sweep finishes.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$HOME/asyncmesh-archives}"
STAMP="$(date +%Y%m%d-%H%M%S)"
DEST="$ARCHIVE_ROOT/$STAMP"
mkdir -p "$DEST"

cd "$REPO"
echo "repo    : $REPO"
echo "archive : $DEST"

{
    echo "archived    : $(date -Is)"
    echo "host        : $(hostname)"
    echo "repo        : $REPO"
    echo "commit      : $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
    echo "branch      : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')"
    echo
    echo "--- uncommitted changes at archive time ---"
    git status --short 2>/dev/null || true
    echo
    echo "--- result JSONs ---"
    find results -name '*.json' 2>/dev/null | wc -l
    echo
    echo "--- W&B offline runs ---"
    find wandb -maxdepth 1 -type d -name 'offline-run-*' 2>/dev/null | wc -l
    echo "(a run is SYNCED once it contains a .wandb.synced marker)"
    find wandb -maxdepth 1 -type d -name 'offline-run-*' 2>/dev/null | while read -r r; do
        if [[ -f "$r/.wandb.synced" ]]; then echo "  synced   $(basename "$r")";
        else echo "  UNSYNCED $(basename "$r")"; fi
    done
} > "$DEST/MANIFEST.txt" 2>&1

for d in results figures logs wandb; do
    if [[ -d "$REPO/$d" ]] && [[ -n "$(ls -A "$REPO/$d" 2>/dev/null)" ]]; then
        echo "  packing $d ..."
        tar czf "$DEST/$d.tar.gz" "$d"
    else
        echo "  skipping $d (absent or empty)"
    fi
done

echo
echo "wrote:"
ls -lh "$DEST"
echo
echo "manifest:"
sed -n '1,12p' "$DEST/MANIFEST.txt"
