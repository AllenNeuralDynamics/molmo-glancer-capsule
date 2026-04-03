#!/usr/bin/env bash
# stash_results.sh — Move /results/ contents to a named stash in /scratch/results_stash/
# Usage: bash /code/stash_results.sh [name]
#   If no name given, uses a timestamp.

set -euo pipefail

STASH_DIR="/scratch/results_stash"
NAME="${1:-$(date +%Y%m%d_%H%M%S)}"
DEST="$STASH_DIR/$NAME"

if [ ! -d /results ] || [ -z "$(ls -A /results 2>/dev/null)" ]; then
    echo "Nothing to stash — /results/ is empty."
    exit 0
fi

if [ -d "$DEST" ]; then
    echo "Stash '$NAME' already exists at $DEST"
    exit 1
fi

mkdir -p "$DEST"
mv /results/* "$DEST/"
echo "Stashed /results/ → $DEST/ ($(ls "$DEST" | wc -l) items)"
