#!/usr/bin/env python3
"""Print recent gateway request traces from Jaeger as span trees, with the
service and duration of every span: the per-service latency breakdown in text
form (the same data as the Jaeger UI at http://localhost:16686).

Usage: python3 scripts/trace_tree.py [-n 2] [--jaeger http://localhost:16686] [--cache hit|miss|coalesced|bypass]
Stdlib only. Uses Jaeger's v3 query API.
"""
import argparse
import datetime as dt
import json
import urllib.parse
import urllib.request

ROOT = "POST /v1/chat/completions"


def fetch_traces(jaeger: str, depth: int) -> list[dict]:
    now = dt.datetime.now(dt.timezone.utc)
    q = urllib.parse.urlencode({
        "query.service_name": "gateway",
        "query.operation_name": ROOT,
        "query.start_time_min": (now - dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query.start_time_max": (now + dt.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query.search_depth": depth,
    })
    with urllib.request.urlopen(f"{jaeger}/api/v3/traces?{q}", timeout=10) as res:
        body = json.load(res)

    # OTLP JSON: resourceSpans[].scopeSpans[].spans[]; group spans by trace.
    traces: dict[str, list[dict]] = {}
    for rs in body.get("result", {}).get("resourceSpans", []):
        service = next(
            (a["value"]["stringValue"] for a in rs["resource"]["attributes"] if a["key"] == "service.name"), "?"
        )
        for ss in rs.get("scopeSpans", []):
            for s in ss.get("spans", []):
                s["service"] = service
                s["attrs"] = {a["key"]: next(iter(a["value"].values())) for a in s.get("attributes", [])}
                traces.setdefault(s["traceId"], []).append(s)
    return list(traces.values())


def ms(span: dict) -> float:
    return (int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"])) / 1e6


def print_tree(spans: list[dict]) -> None:
    ids = {s["spanId"] for s in spans}
    children: dict[str | None, list[dict]] = {}
    for s in spans:
        parent = s.get("parentSpanId")
        children.setdefault(parent if parent in ids else None, []).append(s)
    root = next(s for s in children[None] if s["name"] == ROOT)
    services = sorted({s["service"] for s in spans})
    print(f"\ntrace {root['traceId'][:16]}  cache={root['attrs'].get('cache')}  "
          f"total={ms(root):.1f}ms  spans={len(spans)}  services={services}")

    def walk(s: dict, depth: int) -> None:
        print(f"  {'  ' * depth}{s['name'][:58]:<{60 - 2 * depth}}{s['service']:<17}{ms(s):>8.1f}ms")
        for c in sorted(children.get(s["spanId"], []), key=lambda c: int(c["startTimeUnixNano"])):
            walk(c, depth + 1)

    walk(root, 0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=2, help="how many traces to print")
    p.add_argument("--jaeger", default="http://localhost:16686")
    p.add_argument("--cache", help="only traces with this cache outcome")
    args = p.parse_args()

    traces = []
    for spans in fetch_traces(args.jaeger, depth=max(50, args.n * 5)):
        root = next((s for s in spans if s["name"] == ROOT), None)
        if root and (args.cache is None or root["attrs"].get("cache") == args.cache):
            traces.append((int(root["startTimeUnixNano"]), spans))
    traces.sort(key=lambda t: t[0])
    for _, spans in traces[-args.n:]:
        print_tree(spans)


if __name__ == "__main__":
    main()
