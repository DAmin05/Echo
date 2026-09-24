#!/usr/bin/env python3
"""Phase 1 demo: send groups of paraphrased prompts through the gateway and
report which ones the semantic cache caught.

Within each group the first prompt should miss (it seeds the cache) and the
paraphrases should hit. The unrelated prompts at the end should all miss.

Usage: python3 scripts/phase1_demo.py [--url http://localhost:8080] [--model claude-haiku-4-5]
Stdlib only, no pip install needed.
"""
import argparse
import json
import time
import urllib.error
import urllib.request

GROUPS = [
    [
        "What is the capital of France?",
        "What's the capital city of France?",
        "Tell me the capital of France.",
        "Which city is France's capital?",
    ],
    [
        "How do I reverse a list in Python?",
        "What's the way to reverse a Python list?",
        "Reverse a list in python - how?",
    ],
    [
        "Explain what a mutex is in one sentence.",
        "In one sentence, explain what a mutex is.",
        "Give me a one-sentence explanation of a mutex.",
    ],
    [
        "What is the boiling point of water at sea level?",
        "At sea level, what temperature does water boil at?",
    ],
]
UNRELATED = [
    "Name three moons of Jupiter.",
    "What does HTTP status code 418 mean?",
    "Who wrote Pride and Prejudice?",
]


def ask(url: str, model: str, prompt: str) -> tuple[str, str, float, str]:
    body = json.dumps({
        "model": model,
        # Identical system prompt on every request: it's part of the cache key,
        # so it must match exactly for paraphrases to hit.
        "messages": [
            {"role": "system", "content": "Answer in one short sentence."},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body, headers={"content-type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            data = json.load(res)
            headers = res.headers
    except urllib.error.HTTPError as e:
        raise SystemExit(f"gateway returned {e.code}: {e.read().decode()}")
    ms = (time.perf_counter() - start) * 1000
    answer = data["choices"][0]["message"]["content"].strip().replace("\n", " ")
    return headers.get("x-echo-cache", "?"), headers.get("x-echo-similarity", ""), ms, answer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--model", default="claude-haiku-4-5")
    args = p.parse_args()

    latencies: dict[str, list[float]] = {"hit": [], "miss": []}
    print(f"{'cache':<6} {'sim':<7} {'ms':>7}  prompt -> answer")
    for group in GROUPS + [[u] for u in UNRELATED]:
        for prompt in group:
            status, sim, ms, answer = ask(args.url, args.model, prompt)
            latencies.setdefault(status, []).append(ms)
            print(f"{status:<6} {sim:<7} {ms:7.0f}  {prompt!r} -> {answer[:60]!r}")
        print()

    with urllib.request.urlopen(f"{args.url}/stats") as res:
        print("gateway /stats (all-time):", json.load(res))
    for kind in ("hit", "miss"):
        if latencies[kind]:
            vals = sorted(latencies[kind])
            print(f"{kind:>4}: n={len(vals)}  median={vals[len(vals) // 2]:.0f}ms")


if __name__ == "__main__":
    main()
