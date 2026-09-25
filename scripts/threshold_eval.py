#!/usr/bin/env python3
"""Phase 3: choose the matching thresholds from data.

Simulates the cache holding every anchor question in eval/paraphrases.json,
then asks two kinds of queries, using the running embedding-svc's models:

  - paraphrases (same question, different wording): should HIT their own anchor.
  - hard negatives (similar wording, different question): should MISS. Any hit
    serves the answer to a different question.

Two matching strategies are evaluated, mirroring the gateway:

  similarity only   nearest anchor, accepted if cosine >= threshold
                    (VERIFY_ENABLED=false)
  verified          top-3 anchors with cosine >= candidate threshold, each scored
                    by the cross-encoder; best score >= verify threshold wins
                    (the default)

Metrics:
  paraphrase hit rate   paraphrases served their own anchor's answer (savings)
  wrong answers         hits that returned the answer to a different question
  precision             correct hits / all hits

No LLM calls. Usage:
  python3 scripts/threshold_eval.py [--embed-url http://localhost:8001] [--show 8]
"""
import argparse
import json
import math
import urllib.request
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "eval" / "paraphrases.json"
SIMILARITY_THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97]
CANDIDATE_THRESHOLDS = [0.60, 0.70, 0.80]
VERIFY_THRESHOLDS = [0.50, 0.70, 0.80, 0.90]
TOP_K = 3  # gateway's CACHE_CANDIDATES


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.load(res)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def row(label: str, correct: int, wrong: int, total: int) -> str:
    hits = correct + wrong
    precision = correct / hits if hits else 1.0
    return f"{label}  {correct:>3}/{total} = {correct / total:>5.1%}  {wrong:>6}  {precision:>9.1%}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embed-url", default="http://localhost:8001")
    p.add_argument("--show", type=int, default=6, help="how many borderline cases to list")
    args = p.parse_args()
    embed = lambda text: post(f"{args.embed_url}/embed", {"text": text})["vector"]
    verify = lambda query, cands: post(f"{args.embed_url}/score-duplicates", {"query": query, "candidates": cands})["scores"]

    items = json.loads(DATASET.read_text())
    anchors = [i["anchor"] for i in items]
    queries = [(k, text, True) for k, i in enumerate(items) for text in i["paraphrases"]]
    queries += [(k, i["negative"], False) for k, i in enumerate(items)]
    n_para = sum(1 for q in queries if q[2])
    print(f"Scoring {len(anchors)} anchors against {n_para} paraphrases and {len(queries) - n_para} hard negatives...")

    anchor_vecs = [embed(a) for a in anchors]
    # Per query: top-K anchors as (anchor index, cosine, cross-encoder score).
    scored = []
    for own, text, is_para in queries:
        v = embed(text)
        top = sorted(((j, cosine(v, a)) for j, a in enumerate(anchor_vecs)), key=lambda r: -r[1])[:TOP_K]
        ce = verify(text, [anchors[j] for j, _ in top])
        scored.append((own, text, is_para, [(j, s, x) for (j, s), x in zip(top, ce)]))

    def evaluate(accept) -> tuple[int, int]:
        correct = wrong = 0
        for own, _, is_para, top in scored:
            chosen = accept(top)
            if chosen is None:
                continue
            if is_para and chosen == own:
                correct += 1
            else:
                wrong += 1
        return correct, wrong

    header = f"{'paraphrase hit rate':>21}  {'wrong':>6}  {'precision':>9}"
    print(f"\nSimilarity only (VERIFY_ENABLED=false)\n{'threshold':>9}  {header}")
    for t in SIMILARITY_THRESHOLDS:
        c, w = evaluate(lambda top: top[0][0] if top[0][1] >= t else None)
        print(row(f"{t:>9.2f}", c, w, n_para))

    print(f"\nVerified (top-{TOP_K} candidates, cross-encoder)\n{'candidate':>9} {'verify':>6}  {header}")
    for ct in CANDIDATE_THRESHOLDS:
        for vt in VERIFY_THRESHOLDS:
            def accept(top, ct=ct, vt=vt):
                ok = [(j, x) for j, s, x in top if s >= ct and x >= vt]
                return max(ok, key=lambda r: r[1])[0] if ok else None
            c, w = evaluate(accept)
            print(row(f"{ct:>9.2f} {vt:>6.2f}", c, w, n_para))

    negs = sorted(((top[0][2], top[0][1], text, anchors[top[0][0]]) for _, text, is_para, top in scored if not is_para), reverse=True)
    print(f"\nHard negatives the verifier found most convincing (verifier score, cosine):")
    for x, s, text, anchor in negs[: args.show]:
        print(f"  {x:.3f}  {s:.3f}  {text!r} -> {anchor!r}")


if __name__ == "__main__":
    main()
