#!/bin/bash
set -e
cd "$(dirname "$0")/.."
CKPT="${1:-checkpoints/safe_cql_limit0.1_seed42.pt}"
python scripts/evaluate_mismatch.py \
  --checkpoint "$CKPT" \
  -o results/qc_mismatch_elderly.csv \
  --n-ep 100 \
  --cohort-id elderly_frail
python scripts/plot_mismatch_beautiful.py \
  -i results/qc_mismatch_elderly.csv \
  -o figures/beautiful_mismatch_plot.png
echo "OK: results/qc_mismatch_elderly.csv -> figures/beautiful_mismatch_plot.png"
