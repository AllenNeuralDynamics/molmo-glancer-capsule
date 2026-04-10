#!/usr/bin/env bash
# run.sh — Entry point for molmo-glancer agent pipeline.
#
# Usage:
#   bash run.sh --preset neurons        # single preset
#   bash run.sh --preset all            # all presets with stashing
#
# When --preset all:
#   1. Runs dev startup
#   2. Runs each preset, stashing results after each
#   3. Unstashes all into /results/<preset_name>/
#   4. Prints summary table

set -euo pipefail

export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/scratch/ms-playwright}"
export HF_HOME="${HF_HOME:-/scratch/huggingface}"
export NEUROGLANCER_BASE="${NEUROGLANCER_BASE:-https://neuroglancer-demo.appspot.com}"

RESULTS_DIR="${RESULTS_DIR:-/results}"
mkdir -p "$RESULTS_DIR"

echo "=== molmo-glancer ==="
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start: $(date -Iseconds)"
echo ""

# ── Check if --preset all ────────────────────────────────────────────────
PRESET_ARG=""
for arg in "$@"; do
    if [[ "$PRESET_ARG" == "pending" ]]; then
        PRESET_ARG="$arg"
        break
    fi
    if [[ "$arg" == "--preset" ]]; then
        PRESET_ARG="pending"
    fi
done

if [[ "$PRESET_ARG" == "all" ]]; then
    # ── All-presets mode ─────────────────────────────────────────────────
    PRESETS=(neurons alignment neurons_large alignment_loop segmentation)
    TOTAL_START=$(date +%s)
    declare -a DURATIONS
    declare -a STATUSES

    echo "========================================================"
    echo "  All-Presets Run: ${PRESETS[*]}"
    echo "========================================================"

    # Phase 1: Dev startup
    echo ""
    echo "  Phase 1/3: Dev startup"
    bash /code/_dev_startup.sh

    # Phase 2: Run each preset → stash
    echo ""
    echo "  Phase 2/3: Running ${#PRESETS[@]} presets"

    for i in "${!PRESETS[@]}"; do
        preset="${PRESETS[$i]}"
        n=$((i + 1))
        echo ""
        echo "────────────────────────────────────────────────────"
        echo "  Preset ${n}/${#PRESETS[@]}: ${preset}"
        echo "  Started: $(date -Iseconds)"
        echo "────────────────────────────────────────────────────"

        preset_start=$(date +%s)
        bash /code/cleanup.sh

        if python3 -u /code/molmo_glancer.py --preset "$preset" 2>&1 | tee "$RESULTS_DIR/output.log"; then
            STATUSES[$i]="PASS"
        else
            STATUSES[$i]="FAIL"
            echo ""
            echo "  *** PRESET FAILED: ${preset} ***"
        fi

        bash /code/stash_results.sh "$preset" || true

        preset_end=$(date +%s)
        DURATIONS[$i]=$(( preset_end - preset_start ))
        echo "  Preset ${preset}: ${STATUSES[$i]} (${DURATIONS[$i]}s)"
    done

    # Phase 3: Unstash all
    echo ""
    echo "  Phase 3/3: Unstash all results"
    bash /code/cleanup.sh
    bash /code/unstash_results.sh

    # Summary
    TOTAL_END=$(date +%s)
    TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

    echo ""
    echo "========================================================"
    echo "  SUMMARY"
    echo "========================================================"
    printf "  %-20s %-8s %s\n" "PRESET" "STATUS" "DURATION"
    printf "  %-20s %-8s %s\n" "-------" "------" "--------"
    for i in "${!PRESETS[@]}"; do
        printf "  %-20s %-8s %ss\n" "${PRESETS[$i]}" "${STATUSES[$i]}" "${DURATIONS[$i]}"
    done
    echo "  ────────────────────────────────────────"
    printf "  %-20s %-8s %ss\n" "TOTAL" "" "$TOTAL_ELAPSED"
    echo ""
    echo "  Results: /results/"
    echo "  Finished: $(date -Iseconds)"
    echo "========================================================"
else
    # ── Single-preset mode ───────────────────────────────────────────────
    python3 -u /code/molmo_glancer.py "$@" 2>&1 | tee "$RESULTS_DIR/output.log"
fi
