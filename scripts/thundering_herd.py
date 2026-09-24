#!/usr/bin/env python3
"""Phase 3 demo: fire a burst of concurrent, near-identical requests at the
gateway and count how many reached the LLM.

Every run uses a fresh system prompt (it's part of the cache key), so the
burst always starts cold: nothing is cached, and without in-flight dedup
every request would call the provider.

Usage: python3 scripts/thundering_herd.py [-n 20] [--url http://localhost:8080] [--model claude-haiku-4-5]
Stdlib only.
"""
import argparse
import json
import threading
import time
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

PARAPHRASES = [
    "What is the capital of Australia?",
    "What's the capital city of Australia?",
    "Which city is Australia's capital?",
    "Tell me the capital of Australia.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=20, help="concurrent requests")
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--model", default="claude-haiku-4-5")
    args = p.parse_args()

    system = f"Answer in one short sentence. (burst {uuid.uuid4().hex[:8]})"
    start_gate = threading.Barrier(args.n)

    def fire(i: int) -> tuple[str, float, str]:
        body = json.dumps({
            "model": args.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": PARAPHRASES[i % len(PARAPHRASES)]},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{args.url}/v1/chat/completions", data=body, headers={"content-type": "application/json"}
        )
        start_gate.wait()  # release all requests at the same instant
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as res:
            answer = json.load(res)["choices"][0]["message"]["content"]
            return res.headers.get("x-echo-cache", "?"), (time.perf_counter() - t) * 1000, answer

    def warm(_: int) -> None:
        start_gate.wait()
        urllib.request.urlopen(f"{args.url}/healthz", timeout=10).read()

    with ThreadPoolExecutor(max_workers=args.n) as pool:
        # Docker Desktop's port forwarding takes ~1-2s to accept the first
        # burst of concurrent connections after idling. Absorb that here so
        # the timings below measure Echo, not the port forwarder.
        list(pool.map(warm, range(args.n)))
        results = list(pool.map(fire, range(args.n)))

    counts = Counter(status for status, _, _ in results)
    llm_calls = counts["miss"] + counts["bypass"]
    print(f"{args.n} concurrent requests ({len(PARAPHRASES)} paraphrases of one question)")
    for status, n in counts.most_common():
        print(f"  {status:<10} {n}")
    print(f"  LLM calls  {llm_calls}")
    print(f"  distinct answers: {len({a for _, _, a in results})}")
    lat = sorted(ms for _, ms, _ in results)
    print(f"  latency: min {lat[0]:.0f}ms, median {lat[len(lat) // 2]:.0f}ms, max {lat[-1]:.0f}ms")


if __name__ == "__main__":
    main()
