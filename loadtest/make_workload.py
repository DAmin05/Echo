#!/usr/bin/env python3
"""Generates loadtest/workload.json: a shuffled, mixed stream of chat prompts.

  exact       40%  repeats of the 50 "popular" questions (eval/paraphrases.json
                   anchors), Zipf-weighted so a few are asked far more often,
                   like real traffic
  paraphrase  30%  rewordings of those questions (the eval paraphrases)
  unique      30%  questions asked exactly once. Deliberately look-alike
                   ("What is the capital of Peru?" / "...of Chile?"), so they
                   double as a correctness check: any hit or coalesce on a
                   unique prompt served the answer to a different question.

Deterministic for a given --seed. Usage:
  python3 loadtest/make_workload.py [-n 1000] [--seed 7]
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval" / "paraphrases.json"

# Entities avoid anything the popular questions already cover (France,
# Germany, Japan, China, ...), so a unique prompt never has a legitimate match.
COUNTRIES = [
    "Peru", "Chile", "Kenya", "Norway", "Vietnam", "Egypt", "Portugal", "Argentina", "Thailand", "Poland",
    "Morocco", "Finland", "Colombia", "Ireland", "Nepal", "Hungary", "Ghana", "New Zealand", "Greece", "Cuba",
    "Iceland", "Mongolia", "Uruguay", "Croatia", "Senegal", "Jordan", "Bolivia", "Latvia", "Tunisia", "Laos",
]
CITIES = [
    "Lima", "Nairobi", "Oslo", "Hanoi", "Lisbon", "Bogotá", "Dublin", "Kathmandu", "Budapest", "Accra",
    "Athens", "Havana", "Reykjavik", "Montevideo", "Zagreb", "Dakar", "Amman", "La Paz", "Riga", "Tunis",
]
BOOKS = [
    "Dune", "Moby-Dick", "Jane Eyre", "Brave New World", "The Hobbit", "Beloved", "Dracula", "Middlemarch",
    "The Odyssey", "Frankenstein", "Things Fall Apart", "Don Quixote", "The Trial", "Rebecca", "Emma",
    "Ulysses", "Persuasion", "Catch-22", "The Road", "Lolita",
]
ELEMENTS = [
    "helium", "lithium", "neon", "sodium", "magnesium", "aluminum", "silicon", "phosphorus", "sulfur", "argon",
    "potassium", "calcium", "iron", "copper", "zinc", "tin", "iodine", "platinum", "mercury", "lead",
]
PAINTINGS = [
    "The Last Supper", "The Scream", "Guernica", "The Night Watch", "Girl with a Pearl Earring",
    "The Persistence of Memory", "The Birth of Venus", "American Gothic", "The Kiss", "Las Meninas",
    "Water Lilies", "The Garden of Earthly Delights", "Nighthawks", "The Arnolfini Portrait", "Liberty Leading the People",
    "The Great Wave off Kanagawa", "A Sunday Afternoon on the Island of La Grande Jatte", "The Hay Wain", "Olympia", "The School of Athens",
]
ANIMALS = [
    "tortoise", "parrot", "elephant", "house cat", "horse", "goldfish", "blue whale", "koala", "rabbit", "bald eagle",
    "octopus", "honeybee queen", "giraffe", "penguin", "hamster", "crocodile", "gorilla", "flamingo", "dolphin", "red fox",
]
TEMPLATES = [
    ("What is the largest city in {}?", COUNTRIES),
    ("What is the population of {}?", CITIES),
    ("What is the main theme of {}?", BOOKS),
    ("Who painted {}?", PAINTINGS),
    ("How long does a {} typically live?", ANIMALS),
    ("What is the capital of {}?", COUNTRIES),
    ("What is the official language of {}?", COUNTRIES),
    ("What is the currency of {}?", COUNTRIES),
    ("What country is {} in?", CITIES),
    ("What is {} best known for?", CITIES),
    ("Who wrote {}?", BOOKS),
    ("In what year was {} first published?", BOOKS),
    ("What is the chemical symbol for {}?", ELEMENTS),
    ("What is the atomic number of {}?", ELEMENTS),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.seed)

    items = json.loads(EVAL.read_text())
    uniques = [t.format(x) for t, xs in TEMPLATES for x in xs]
    rng.shuffle(uniques)

    n_exact, n_para = int(args.n * 0.4), int(args.n * 0.3)
    n_unique = args.n - n_exact - n_para
    if n_unique > len(uniques):
        raise SystemExit(f"only {len(uniques)} unique prompts available; lower -n")

    # Zipf(s=1) popularity over the 50 popular questions.
    weights = [1 / (rank + 1) for rank in range(len(items))]
    popular_order = list(range(len(items)))
    rng.shuffle(popular_order)  # which question is most popular is arbitrary

    workload = []
    for _ in range(n_exact):
        g = rng.choices(popular_order, weights)[0]
        workload.append({"category": "exact", "group": g, "prompt": items[g]["anchor"]})
    for _ in range(n_para):
        g = rng.choices(popular_order, weights)[0]
        workload.append({"category": "paraphrase", "group": g, "prompt": rng.choice(items[g]["paraphrases"])})
    for prompt in uniques[:n_unique]:
        workload.append({"category": "unique", "group": None, "prompt": prompt})
    rng.shuffle(workload)

    out = HERE / "workload.json"
    out.write_text(json.dumps(workload, indent=0))
    counts = {c: sum(1 for w in workload if w["category"] == c) for c in ("exact", "paraphrase", "unique")}
    print(f"wrote {len(workload)} requests to {out.relative_to(HERE.parent)}: {counts}, "
          f"{len({w['prompt'] for w in workload})} distinct prompts")


if __name__ == "__main__":
    main()
