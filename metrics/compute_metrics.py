"""
Computes two metrics on a saved orchestrator run:
1. Grounding rate -- does each persona's reaction reference real,
   injected facts from the ward context (flood location, flooding
   terms, BBMP, bylaw/regulatory terms)?
2. Distinctiveness -- pairwise TF-IDF cosine similarity between
   persona reactions (lower = more distinct).

This version uses only Python's built-in modules (math, re,
collections) -- no scikit-learn/numpy needed, to avoid Mac
architecture install issues (the same problem we hit earlier with
pandas). The TF-IDF and cosine similarity math is implemented
directly; it produces the same kind of result sklearn's
TfidfVectorizer + cosine_similarity would, just without the
dependency.

Usage:
    python3 metrics/compute_metrics.py outputs/run_20260831_210628.json
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

# A small, standard English stopword list -- enough to match sklearn's
# default behaviour closely for this purpose, without needing the package.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
    "in", "into", "is", "it", "no", "not", "of", "on", "or", "such",
    "that", "the", "their", "then", "there", "these", "they", "this",
    "to", "was", "will", "with", "we", "i", "you", "your", "my", "our",
    "would", "could", "should", "can", "just", "about", "also", "its",
    "it's", "don't", "doesn't", "isn't", "wasn't", "has", "have", "had",
    "them", "us", "me", "do", "does", "did", "so", "than", "too", "very",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


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
        "overall_rate_pct": round(100 * total_hits / total_possible, 1),
        "total_hits": total_hits,
        "total_possible": total_possible,
    }


def compute_tfidf_vectors(texts: list[str]) -> list[dict]:
    """Manual TF-IDF: term frequency * inverse document frequency."""
    tokenized = [tokenize(t) for t in texts]
    n_docs = len(tokenized)

    # Document frequency: how many documents each term appears in
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


def distinctiveness(reactions: dict[str, str]) -> dict:
    persona_ids = list(reactions.keys())
    texts = [reactions[pid] for pid in persona_ids]
    vectors = compute_tfidf_vectors(texts)

    pairs = {}
    sims = []
    for i in range(len(persona_ids)):
        for j in range(i + 1, len(persona_ids)):
            key = f"{persona_ids[i]}-{persona_ids[j]}"
            sim = round(cosine_similarity(vectors[i], vectors[j]), 3)
            pairs[key] = sim
            sims.append(sim)

    return {
        "pairwise_similarity": pairs,
        "average_similarity": round(sum(sims) / len(sims), 3) if sims else None,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 metrics/compute_metrics.py <path_to_run.json>")
        sys.exit(1)

    run_path = Path(sys.argv[1])
    with open(run_path) as f:
        run = json.load(f)

    reactions = {r["persona_id"]: r["reaction"] for r in run["reactions"]}

    g = grounding_rate(reactions)
    d = distinctiveness(reactions)

    print(f"\n=== Grounding rate (run: {run_path.name}) ===")
    for pid, info in g["per_persona"].items():
        print(f"  {pid}: {info['count']}/{info['out_of']} categories "
              f"({', '.join(info['categories_hit']) or 'none'})")
    print(f"  Overall: {g['total_hits']}/{g['total_possible']} "
          f"= {g['overall_rate_pct']}%")

    print(f"\n=== Distinctiveness (pairwise TF-IDF cosine similarity) ===")
    for pair, sim in d["pairwise_similarity"].items():
        print(f"  {pair}: {sim}")
    print(f"  Average: {d['average_similarity']}")

    out_path = run_path.parent / f"metrics_{run_path.stem}.json"
    with open(out_path, "w") as f:
        json.dump({"grounding": g, "distinctiveness": d}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
