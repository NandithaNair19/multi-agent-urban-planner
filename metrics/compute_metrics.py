"""
Metrics for saved orchestrator runs (outputs/run_*.json).

Three measurements, all computed with built-in Python only (no
scikit-learn / numpy, to avoid the Apple Silicon install problems):

1. Grounding rate: for each persona's reaction, how many of 4 fact
   categories it mentions (flood location, flooding terms, BBMP,
   bylaw/regulatory terms). Keyword matching.
2. Distinctiveness: pairwise TF-IDF cosine similarity between
   different personas' reactions in the same run (lower = more
   distinct).
3. Consistency (multi-run mode only): pairwise TF-IDF cosine
   similarity between the SAME persona's reactions across runs.
   This measures wording overlap only. It does not prove the
   persona's stance is stable; read the reactions to check that.

Usage:
    One run:    python3 metrics/compute_metrics.py outputs/run_20260831_210628.json
    Many runs:  python3 metrics/compute_metrics.py outputs/run_*.json

With several files, each run is scored as before, then an aggregate
summary (means, ranges, consistency) is printed and saved to
metrics_summary.json next to the first run file.

To change what counts as a "grounding" fact, edit GROUNDING_TERMS.
"""
import json
import math
import re
import sys
from pathlib import Path
from collections import Counter

GROUNDING_TERMS = {
    "location": ["sony signal", "maharaja signal", "sony-maharaja", "sony world"],
    "flood": ["flood", "monsoon", "stormwater", "drain"],
    "authority": ["bbmp"],
    "regulatory": ["bda", "ktcp", "sanction", "land-use", "land use", "notified"],
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
    "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
    "that", "the", "their", "then", "there", "these", "they", "this",
    "to", "was", "will", "with", "we", "i", "you", "your", "my", "our",
    "would", "could", "should", "can", "just", "about", "also", "its",
    "it's", "don't", "doesn't", "isn't", "wasn't", "has", "have", "had",
    "them", "us", "me", "do", "does", "did", "so", "than", "too", "very",
}


# ---------- helpers ----------

def round_half_up(x: float, ndigits: int) -> float:
    """Ordinary rounding (81.25 -> 81.3). Python's round() would give 81.2."""
    f = 10 ** ndigits
    return math.floor(x * f + 0.5) / f


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def compute_tfidf_vectors(texts: list[str]) -> list[dict]:
    """Manual TF-IDF: term frequency * inverse document frequency."""
    tokenized = [tokenize(t) for t in texts]
    n_docs = len(tokenized)

    doc_freq = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            doc_freq[term] += 1

    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total_terms = len(tokens) or 1
        vec = {}
        for term, count in tf.items():
            tf_score = count / total_terms
            idf_score = math.log((n_docs + 1) / (doc_freq[term] + 1)) + 1
            vec[term] = tf_score * idf_score
        vectors.append(vec)
    return vectors


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    common_terms = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[t] * vec_b[t] for t in common_terms)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------- single-run metrics ----------

def grounding_rate(reactions: dict[str, str]) -> dict:
    per_persona = {}
    total_hits = 0
    total_possible = len(reactions) * len(GROUNDING_TERMS)

    for persona_id, text in reactions.items():
        text_l = text.lower()
        hits = [
            category for category, terms in GROUNDING_TERMS.items()
            if any(term in text_l for term in terms)
        ]
        per_persona[persona_id] = {
            "categories_hit": hits,
            "count": len(hits),
            "out_of": len(GROUNDING_TERMS),
        }
        total_hits += len(hits)

    return {
        "per_persona": per_persona,
        "overall_rate_pct": round_half_up(100 * total_hits / total_possible, 1),
        "total_hits": total_hits,
        "total_possible": total_possible,
    }


def distinctiveness(reactions: dict[str, str]) -> dict:
    persona_ids = list(reactions.keys())
    texts = [reactions[pid] for pid in persona_ids]
    vectors = compute_tfidf_vectors(texts)

    pairs, pairs_raw, sims = {}, {}, []
    for i in range(len(persona_ids)):
        for j in range(i + 1, len(persona_ids)):
            key = f"{persona_ids[i]}-{persona_ids[j]}"
            sim = cosine_similarity(vectors[i], vectors[j])
            pairs[key] = round_half_up(sim, 3)
            pairs_raw[key] = sim  # unrounded, used when averaging across runs
            sims.append(sim)

    return {
        "pairwise_similarity": pairs,
        "pairwise_similarity_raw": pairs_raw,
        "average_similarity": round_half_up(mean(sims), 3) if sims else None,
    }


def load_reactions(run_path: Path) -> dict[str, str]:
    with open(run_path) as f:
        run = json.load(f)
    return {r["persona_id"]: r["reaction"] for r in run["reactions"]}


def print_single_run(name: str, g: dict, d: dict) -> None:
    print(f"\n=== Grounding rate (run: {name}) ===")
    for pid, info in g["per_persona"].items():
        print(f"  {pid}: {info['count']}/{info['out_of']} categories "
              f"({', '.join(info['categories_hit']) or 'none'})")
    print(f"  Overall: {g['total_hits']}/{g['total_possible']} "
          f"= {g['overall_rate_pct']}%")

    print("\n=== Distinctiveness (pairwise TF-IDF cosine similarity) ===")
    for pair, sim in d["pairwise_similarity"].items():
        print(f"  {pair}: {sim}")
    print(f"  Average: {d['average_similarity']}")


# ---------- multi-run aggregation ----------

def pair_key(a: str, b: str, order: list[str]) -> str:
    """Same pair name no matter which persona was listed first in a run."""
    a, b = sorted([a, b], key=lambda p: order.index(p))
    return f"{a}-{b}"


def aggregate(run_names: list[str], per_run: list[dict]) -> dict:
    """per_run items: {"reactions", "grounding", "distinctiveness"}"""
    order = []
    for item in per_run:
        for pid in item["reactions"]:
            if pid not in order:
                order.append(pid)

    # grounding per persona
    grounding = {}
    for pid in order:
        counts = [it["grounding"]["per_persona"][pid]["count"]
                  for it in per_run if pid in it["grounding"]["per_persona"]]
        out_of = len(GROUNDING_TERMS)
        grounding[pid] = {
            "runs": len(counts),
            "categories_per_run": counts,
            "mean_rate_pct": round_half_up(mean([100 * c / out_of for c in counts]), 1),
            "min_categories": min(counts),
            "max_categories": max(counts),
            "out_of": out_of,
        }

    pooled_hits = sum(it["grounding"]["total_hits"] for it in per_run)
    pooled_possible = sum(it["grounding"]["total_possible"] for it in per_run)

    # distinctiveness per pair (unrounded values, normalised pair names)
    pair_values = {}
    for it in per_run:
        for key, sim in it["distinctiveness"]["pairwise_similarity_raw"].items():
            a, b = key.split("-", 1)
            pair_values.setdefault(pair_key(a, b, order), []).append(sim)

    distinct = {
        key: {
            "runs": len(vals),
            "mean": round_half_up(mean(vals), 3),
            "min": round_half_up(min(vals), 3),
            "max": round_half_up(max(vals), 3),
        }
        for key, vals in pair_values.items()
    }
    all_pair_vals = [v for vals in pair_values.values() for v in vals]

    # consistency: same persona across runs
    consistency = {}
    for pid in order:
        texts = [it["reactions"][pid] for it in per_run if pid in it["reactions"]]
        if len(texts) < 2:
            continue
        vecs = compute_tfidf_vectors(texts)
        sims = [cosine_similarity(vecs[i], vecs[j])
                for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        consistency[pid] = {
            "runs": len(texts),
            "pairs": len(sims),
            "mean_similarity": round_half_up(mean(sims), 3),
        }

    return {
        "runs": run_names,
        "n_runs": len(run_names),
        "grounding_per_persona": grounding,
        "grounding_overall": {
            "total_hits": pooled_hits,
            "total_possible": pooled_possible,
            "rate_pct": round_half_up(100 * pooled_hits / pooled_possible, 1),
            "weakest_run_pct": min(it["grounding"]["overall_rate_pct"] for it in per_run),
            "strongest_run_pct": max(it["grounding"]["overall_rate_pct"] for it in per_run),
        },
        "distinctiveness_per_pair": distinct,
        "distinctiveness_overall": {
            "mean": round_half_up(mean(all_pair_vals), 3),
            "max_any_pair_any_run": round_half_up(max(all_pair_vals), 3),
        },
        "consistency_same_persona_across_runs": consistency,
    }


def print_summary(s: dict) -> None:
    print("\n" + "=" * 60)
    print(f"SUMMARY ACROSS {s['n_runs']} RUNS")
    print("=" * 60)
    for name in s["runs"]:
        print(f"  - {name}")

    print("\n--- Mean grounding rate per persona ---")
    for pid, g in s["grounding_per_persona"].items():
        print(f"  {pid}: {g['mean_rate_pct']}%  "
              f"(categories per run: {g['categories_per_run']}, "
              f"range {g['min_categories']}-{g['max_categories']} of {g['out_of']})")
    go = s["grounding_overall"]
    print(f"  Overall: {go['rate_pct']}%  "
          f"({go['total_hits']} of {go['total_possible']} category hits; "
          f"weakest run {go['weakest_run_pct']}%, strongest {go['strongest_run_pct']}%)")

    print("\n--- Mean pairwise distinctiveness (TF-IDF cosine similarity, lower = more distinct) ---")
    for pair, d in s["distinctiveness_per_pair"].items():
        print(f"  {pair}: mean {d['mean']}  (min {d['min']}, max {d['max']}, n={d['runs']} runs)")
    do = s["distinctiveness_overall"]
    print(f"  Overall mean: {do['mean']}   Highest single pair in any run: {do['max_any_pair_any_run']}")

    if s["consistency_same_persona_across_runs"]:
        print("\n--- Consistency: same persona across runs (wording overlap only) ---")
        for pid, c in s["consistency_same_persona_across_runs"].items():
            print(f"  {pid}: mean similarity {c['mean_similarity']}  ({c['pairs']} run pairs)")
        print("  Note: this shows wording overlap, not stance. Read the reactions to judge stance.")


# ---------- main ----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 metrics/compute_metrics.py <run.json> [more run files...]")
        sys.exit(1)

    run_paths = [Path(a) for a in sys.argv[1:]]
    per_run = []

    for run_path in run_paths:
        reactions = load_reactions(run_path)
        g = grounding_rate(reactions)
        d = distinctiveness(reactions)
        print_single_run(run_path.name, g, d)

        out_path = run_path.parent / f"metrics_{run_path.stem}.json"
        with open(out_path, "w") as f:
            json.dump({"grounding": g, "distinctiveness": d}, f, indent=2)
        print(f"\nSaved: {out_path}")

        per_run.append({"reactions": reactions, "grounding": g, "distinctiveness": d})

    if len(run_paths) > 1:
        summary = aggregate([p.name for p in run_paths], per_run)
        print_summary(summary)
        summary_path = run_paths[0].parent / "metrics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()