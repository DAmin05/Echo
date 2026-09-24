#!/usr/bin/env python3
"""Phase 3: choose the similarity threshold from data.

Simulates the cache holding every anchor question in eval/paraphrases.json,
then asks two kinds of queries, embedded by the running embedding-svc:

  - paraphrases (same question, different wording): should HIT their own anchor.
  - hard negatives (similar wording, different question): should MISS. Any hit
    serves the answer to a different question.

For each threshold it reports:
  paraphrase hit rate   paraphrases served their own anchor's answer (savings)
  wrong answers         hits that returned the answer to a different question:
                        negatives that hit anything, plus paraphrases whose
                        nearest anchor was the wrong one
  precision             correct hits / all hits

No LLM calls; only embeddings. Usage:
  python3 scripts/threshold_eval.py [--embed-url http://localhost:8001] [--show 8]
"""
import argparse
import json
import math
import urllib.request
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "eval" / "paraphrases.json"
THRESHOLDS = [0.80, 0.83, 0.85, 0.87, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97]


def embed(url: str, text: str) -> list[float]:
    req = urllib.request.Request(
        f"{url}/embed", data=json.dumps({"text": text}).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.load(res)["vector"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def nearest(vector: list[float], anchors: list[list[float]]) -> tuple[int, float]:
    scores = [cosine(vector, a) for a in anchors]
    best = max(range(len(scores)), key=scores.__getitem__)
    return best, scores[best]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embed-url", default="http://localhost:8001")
    p.add_argument("--show", type=int, default=8, help="how many borderline cases to list")
    args = p.parse_args()

    items = json.loads(DATASET.read_text())
    print(f"Embedding {len(items)} anchors, "
          f"{sum(len(i['paraphrases']) for i in items)} paraphrases, {len(items)} hard negatives...")
    anchors = [embed(args.embed_url, i["anchor"]) for i in items]

    # (own anchor index, text, nearest anchor index, similarity)
    paraphrases = []
    negatives = []
    for idx, item in enumerate(items):
        for text in item["paraphrases"]:
            paraphrases.append((idx, text, *nearest(embed(args.embed_url, text), anchors)))
        negatives.append((idx, item["negative"], *nearest(embed(args.embed_url, item["negative"]), anchors)))

    print()
    print(f"{'threshold':>9}  {'paraphrase hit rate':>19}  {'wrong answers':>13}  {'precision':>9}")
    for t in THRESHOLDS:
        correct = sum(1 for own, _, near, s in paraphrases if s >= t and near == own)
        wrong_para = sum(1 for own, _, near, s in paraphrases if s >= t and near != own)
        wrong_neg = sum(1 for _, _, _, s in negatives if s >= t)
        wrong = wrong_para + wrong_neg
        hits = correct + wrong
        precision = correct / hits if hits else 1.0
        print(f"{t:>9.2f}  {correct:>3}/{len(paraphrases)} = {correct / len(paraphrases):>5.1%}  "
              f"{wrong:>13}  {precision:>9.1%}")

    print(f"\nHighest-scoring hard negatives (would serve the WRONG answer at thresholds below the score):")
    for own, text, near, s in sorted(negatives, key=lambda r: -r[3])[: args.show]:
        print(f"  {s:.4f}  {text!r}  ->  {items[near]['anchor']!r}")

    print(f"\nLowest-scoring paraphrases (missed savings at thresholds above the score):")
    for own, text, near, s in sorted(paraphrases, key=lambda r: r[3])[: args.show]:
        flag = "" if near == own else f"  [nearest is {items[near]['anchor']!r}]"
        print(f"  {s:.4f}  {text!r}  ~  {items[own]['anchor']!r}{flag}")


if __name__ == "__main__":
    main()
