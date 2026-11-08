#!/usr/bin/env bash
# Download the raw Nezha multi-modal RCA dataset (~3.2 GB, MIT licence) and rebuild data/processed.
# The processed bundles are already committed, so this is only needed to regenerate them.
set -euo pipefail
RAW=${1:-../Nezha}
if [ ! -d "$RAW" ]; then
  git clone --depth 1 https://github.com/IntelligentDDS/Nezha "$RAW"
fi
python scripts/preprocess_nezha.py --raw "$RAW" --out data/processed
