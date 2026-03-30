#!/usr/bin/env bash
# cleanup.sh — remove all pipeline artifacts from /results/
set -euo pipefail

rm -rf /results/screenshots
rm -f /results/output.log /results/answer.txt /results/findings.json /results/ng_states.json

echo "Cleaned /results/"
