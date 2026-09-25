#!/usr/bin/env bash
# Runs the k6 load test against the running stack (docker compose up -d),
# then prints the results tables.
#
#   ./loadtest/run.sh                  # 1000 requests at 10 req/s
#   RATE=25 ./loadtest/run.sh
#   WORKLOAD=hits.json RUN_ID=<earlier run> RATE=50 ./loadtest/run.sh   # hit-path capacity (see capacity.sh)
set -euo pipefail
cd "$(dirname "$0")"

[ -f workload.json ] || python3 make_workload.py
mkdir -p results
RUN_ID=${RUN_ID:-$(date +%s)}
OUT=results/raw-${RUN_ID}.json

# On the compose network, so k6 reaches the gateway directly (no host port forwarding).
docker run --rm --network echo_default \
  -v "$PWD":/work -w /work \
  -e RATE="${RATE:-10}" -e MODEL="${MODEL:-claude-haiku-4-5}" -e RUN_ID="$RUN_ID" -e WORKLOAD="${WORKLOAD:-workload.json}" \
  grafana/k6:2.3.0 run --quiet --out "json=$OUT" echo.js

python3 report.py "$OUT" --model "${MODEL:-claude-haiku-4-5}"
