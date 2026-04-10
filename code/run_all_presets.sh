#!/usr/bin/env bash
# run_all_presets.sh — Run all presets end-to-end, stashing results after each.
#
# Usage: bash /code/run_all_presets.sh
#
# Phases:
#   1. Dev startup (install dependencies)
#   2. Download weights (Molmo2 + OLMo)
#   3. Run each preset → stash results
#   4. Unstash all into /results/<preset_name>/

set -euo pipefail

rm -rf /results/*
# rm -rf /scratch/*

PRESETS=(neurons alignment neurons_large alignment_loop segmentation)
TOTAL_START=$(date +%s)

# Per-preset tracking
declare -a DURATIONS
declare -a STATUSES

echo ""
echo "========================================================"
echo "  molmo-glancer — All-Presets Test Run"
echo "  Presets: ${PRESETS[*]}"
echo "  Start:  $(date -Iseconds)"
echo "========================================================"

# ── Phase 1: Dev startup ─────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Phase 1/4: Dev startup"
echo "════════════════════════════════════════════════════════"
# bash /code/_dev_startup.sh

# ── Phase 2: Download weights ────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Phase 2/4: Download weights"
echo "════════════════════════════════════════════════════════"
# bash /code/_download_weights.sh

# ── Phase 3: Run each preset ────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Phase 3/4: Running ${#PRESETS[@]} presets"
echo "════════════════════════════════════════════════════════"

for i in "${!PRESETS[@]}"; do
    preset="${PRESETS[$i]}"
    n=$((i + 1))
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  Preset ${n}/${#PRESETS[@]}: ${preset}"
    echo "  Started: $(date -Iseconds)"
    echo "────────────────────────────────────────────────────"

    preset_start=$(date +%s)

    # Clear results from previous run
    bash /code/cleanup.sh
    RESULTS_DIR="${RESULTS_DIR:-/results}"

    # Run the preset — continue on failure
    if python3 -u /code/molmo_glancer.py --preset "$preset" 2>&1 | tee "$RESULTS_DIR/output.log"; then
        STATUSES[$i]="PASS"
    else
        STATUSES[$i]="FAIL"
        echo ""
        echo "  *** PRESET FAILED: ${preset} ***"
    fi

    # Stash whatever results exist (even partial on failure)
    bash /code/stash_results.sh "$preset" || true

    preset_end=$(date +%s)
    DURATIONS[$i]=$(( preset_end - preset_start ))
    echo ""
    echo "  Preset ${preset}: ${STATUSES[$i]} (${DURATIONS[$i]}s)"
done

# ── Phase 4: Unstash all ─────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Phase 4/4: Unstash all results"
echo "════════════════════════════════════════════════════════"
bash /code/cleanup.sh
bash /code/unstash_results.sh

# ── Summary ──────────────────────────────────────────────────
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
