#!/usr/bin/env bash
# unstash_results.sh — Copy all stashes back into /results/ under named folders.
# Usage: bash /code/unstash_results.sh
#   Each stash becomes /results/<stash_name>/

set -euo pipefail

STASH_DIR="/scratch/results_stash"

if [ ! -d "$STASH_DIR" ] || [ -z "$(ls -A "$STASH_DIR" 2>/dev/null)" ]; then
    echo "No stashes found in $STASH_DIR"
    exit 0
fi

mkdir -p /results

for stash in "$STASH_DIR"/*/; do
    name="$(basename "$stash")"
    dest="/results/$name"
    if [ -d "$dest" ]; then
        echo "  skip: $name (already exists in /results/)"
        continue
    fi
    cp -r "$stash" "$dest"
    echo "  restored: $name → $dest"
done

echo "Done. $(ls -d /results/*/ 2>/dev/null | wc -l) stashes in /results/"
