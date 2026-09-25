#!/usr/bin/env bash
# Hit-path capacity: replays questions already cached by an earlier run
# (same RUN_ID = same system prompt = warm cache) at increasing rates.
# Every request should be a cache hit, so this costs nothing in LLM calls.
#
#   ./loadtest/capacity.sh <RUN_ID of a completed ./run.sh> [rates...]
set -euo pipefail
cd "$(dirname "$0")"
RUN_ID=$1; shift
RATES=${*:-25 50 100}

for rate in $RATES; do
  python3 - "$rate" <<'PY'
import json, random, sys
rate = int(sys.argv[1])
w = json.load(open("workload.json"))
exact = sorted({x["prompt"] for x in w if x["category"] == "exact"})  # all cached by the earlier run
rng = random.Random(rate)
json.dump([{"category": "exact", "group": None, "prompt": rng.choice(exact)} for _ in range(rate * 20)], open("hits.json", "w"))
PY
  echo "=== ${rate} req/s for 20s"
  WORKLOAD=hits.json RUN_ID="$RUN_ID" RATE="$rate" ./run.sh | grep -E "^\| (hit|miss|all) |Served without|dropped" || true
done
