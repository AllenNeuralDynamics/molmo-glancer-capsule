#!/usr/bin/env bash
# cleanup.sh — remove all pipeline artifacts from /results/
set -euo pipefail

rm -rf /results/*

echo "Cleaned /results/"
