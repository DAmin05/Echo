#!/usr/bin/env python3
"""Turns k6's raw JSON output into the results tables for the README.

Usage: python3 loadtest/report.py loadtest/results/raw.json [--model claude-haiku-4-5]

Cost model: every response carries the token usage of the provider call that
produced it (a cache hit replays the original call's usage). So
  cost without Echo = price(all requests' tokens)
  cost with Echo    = price(tokens of requests that actually called the provider)
Projections for other models reuse the same token counts; real counts would
differ somewhat by model.
"""
import argparse
import json
from collections import defaultdict

# $ per million tokens (input, output), Anthropic first-party pricing.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}
CALLED_PROVIDER = {"miss", "bypass"}
SERVED_WITHOUT_PROVIDER = {"hit", "coalesced"}


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, round(q / 100 * (len(s) - 1)))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw")
    p.add_argument("--model", default="claude-haiku-4-5")
    args = p.parse_args()

    lat = defaultdict(list)  # (category, cache) -> [ms]
    tokens = defaultdict(lambda: [0, 0])  # cache -> [input, output]
    for line in open(args.raw):
        point = json.loads(line)
        if point.get("type") != "Point":
            continue
        m, d = point["metric"], point["data"]
        tags = d.get("tags", {})
        if m == "echo_latency":
            lat[(tags["category"], tags["cache"])].append(d["value"])
        elif m == "echo_input_tokens":
            tokens[tags["cache"]][0] += d["value"]
        elif m == "echo_output_tokens":
            tokens[tags["cache"]][1] += d["value"]

    total = sum(len(v) for v in lat.values())
    by_cache = defaultdict(list)
    by_cat = defaultdict(lambda: defaultdict(int))
    for (cat, cache), v in lat.items():
        by_cache[cache] += v
        by_cat[cat][cache] += len(v)

    served = sum(len(by_cache[c]) for c in SERVED_WITHOUT_PROVIDER)
    llm_calls = sum(len(by_cache[c]) for c in CALLED_PROVIDER)
    errors = len(by_cache.get("error", []))
    print(f"## Load test: {total} requests\n")
    print(f"Served without calling the provider: **{served}/{total} = {served / total:.1%}** "
          f"(hit {len(by_cache['hit'])}, coalesced {len(by_cache['coalesced'])}); "
          f"provider calls: {llm_calls}; errors: {errors}\n")

    print("| Category | Requests | Hit | Coalesced | Miss | Bypass | Served from cache |")
    print("|---|---|---|---|---|---|---|")
    for cat in ("exact", "paraphrase", "unique"):
        c = by_cat[cat]
        n = sum(c.values())
        if not n:
            continue
        cached = c["hit"] + c["coalesced"]
        print(f"| {cat} | {n} | {c['hit']} | {c['coalesced']} | {c['miss']} | {c['bypass']} | {cached / n:.1%} |")

    wrong = by_cat["unique"]["hit"] + by_cat["unique"]["coalesced"]
    print(f"\nUnique prompts served from cache (each one a wrong answer): **{wrong}** of {sum(by_cat['unique'].values())}\n")

    print("| Path | Requests | p50 | p95 | p99 |")
    print("|---|---|---|---|---|")
    for cache in ("hit", "coalesced", "miss", "bypass", "error"):
        v = by_cache.get(cache)
        if v:
            print(f"| {cache} | {len(v)} | {pct(v, 50):.0f} ms | {pct(v, 95):.0f} ms | {pct(v, 99):.0f} ms |")
    allv = [x for v in by_cache.values() for x in v]
    print(f"| all | {len(allv)} | {pct(allv, 50):.0f} ms | {pct(allv, 95):.0f} ms | {pct(allv, 99):.0f} ms |")

    tin = sum(t[0] for t in tokens.values())
    tout = sum(t[1] for t in tokens.values())
    pin = sum(tokens[c][0] for c in CALLED_PROVIDER)
    pout = sum(tokens[c][1] for c in CALLED_PROVIDER)
    print(f"\nTokens: {tin:.0f} in / {tout:.0f} out across all responses; "
          f"{pin:.0f} in / {pout:.0f} out actually sent to the provider.\n")
    print("| Model | Cost per 1000 requests without Echo | With Echo | Saved |")
    print("|---|---|---|---|")
    for model, (ci, co) in PRICES.items():
        without = (tin * ci + tout * co) / 1e6 / total * 1000
        with_echo = (pin * ci + pout * co) / 1e6 / total * 1000
        label = model + (" (measured)" if model == args.model else " (projected)")
        saved = 1 - with_echo / without if without else 0
        print(f"| {label} | ${without:.4f} | ${with_echo:.4f} | {saved:.0%} |")


if __name__ == "__main__":
    main()
