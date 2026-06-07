#!/usr/bin/env bash
# Long training run with logging to file. Usage: ./scripts/train_long.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source myenv/bin/activate 2>/dev/null || true
mkdir -p logs
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
python3 train.py --config configs/long_run.yaml 2>&1 | tee "$LOG"
