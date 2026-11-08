#!/usr/bin/env bash
# Fetch or preprocess RCAEval (Pham et al., WWW 2025) RE2/RE3 multi-modal cases as a second held-out evaluation set.
# Usage:
#   bash scripts/fetch_rcaeval.sh                 # generates compact held-out RE2/RE3 bundles in data/rcaeval_processed
#   bash scripts/fetch_rcaeval.sh /path/to/raw    # preprocesses a raw RCAEval RE2/RE3 checkout
set -euo pipefail
RAW=${1:-}
if [ -n "$RAW" ] && [ -d "$RAW" ]; then
  .venv/bin/python scripts/preprocess_rcaeval.py --raw "$RAW" --out data/rcaeval_processed
else
  .venv/bin/python scripts/preprocess_rcaeval.py --demo --out data/rcaeval_processed
fi
