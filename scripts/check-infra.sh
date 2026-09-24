#!/usr/bin/env bash
# Phase 0 check: confirm Qdrant and Redis are up and reachable from the host.
set -euo pipefail

fail=0

echo -n "Qdrant REST (localhost:6333) ... "
if out=$(curl -sf http://localhost:6333/); then
  echo "ok  $(echo "$out" | tr -d '\n' | cut -c1-80)"
else
  echo "FAILED"; fail=1
fi

echo -n "Qdrant collections endpoint ... "
if curl -sf http://localhost:6333/collections >/dev/null; then echo "ok"; else echo "FAILED"; fail=1; fi

echo -n "Qdrant gRPC port (localhost:6334) ... "
if nc -z localhost 6334 2>/dev/null; then echo "ok"; else echo "FAILED"; fail=1; fi

echo -n "Redis (localhost:6379) ... "
if out=$(docker compose exec -T redis redis-cli ping 2>/dev/null) && [ "$out" = "PONG" ]; then
  echo "ok  PONG"
else
  echo "FAILED"; fail=1
fi

echo -n "Redis host port (localhost:6379) ... "
if [ "$(printf 'PING\r\n' | nc -w 2 localhost 6379 | tr -d '\r')" = "+PONG" ]; then echo "ok"; else echo "FAILED"; fail=1; fi

exit $fail
