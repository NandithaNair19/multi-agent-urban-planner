"""
Step 1 of bylaw processing: find CANDIDATE clauses in the BBMP Building
Bye-Laws PDF.

pdfplumber returns page text with no blank lines, so we cannot split on
paragraphs. Instead we split each page at numbered clause headings
(e.g. "7.2 ", "3.10 ", "TABLE 4") and keep every chunk that mentions one
of our topics. Keywords are matched as whole words, so "LIG" does not
match "light" or "religious", and one clause can carry several topics.

This script only FINDS candidates. Choosing which clauses a persona sees
and summarising them is a manual step (data/curated/bylaws_curated.json),
checked against the PDF by data/build_bylaws_summary.py.

Run:    python3 data/parse_bylaws.py
Output: data/processed/bylaw_candidates.json
"""
import json
import re
from pathlib import Path

import pdfplumber

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "processed"
OUT.mkdir(exist_ok=True)

PDF_PATH = RAW / "Bangalore-Building-Byelaws.pdf"

# Topic -> whole-word regex patterns
TOPIC_PATTERNS = {
    "slum_clearance": [r"\bslums?\b"],
    "ews_lig_housing": [r"\bEWS\b", r"\bL\.?I\.?G\.?\b", r"economically weaker"],
    "setbacks": [r"\bset[\s-]?backs?\b"],
    "floor_area_ratio": [r"\bF\.?A\.?R\.?\b", r"floor area ratio"],
    "land_use_change": [r"\bchange of land use\b", r"\bland[\s-]use\b",
                        r"\bsub-?divisions?\b", r"Town and Country Planning"],
}

# A new clause starts at a line like "7.2 No Plan...", "3.10 Exemptions",
# "1.2.6.2 Uses..." or "TABLE 4". Bare page numbers are not headings.
CLAUSE_START = re.compile(r"^(?:\d+(?:\.\d+)+\s+\S|TABLE\s*\d+)", re.MULTILINE)
CLAUSE_ID = re.compile(r"^(\d+(?:\.\d+)+|TABLE\s*\d+)")

# Glossary entries ("'term' means ...") define words but regulate nothing.
DEFINITION_PATTERN = re.compile(r"[‘']\s*[\w\s\-]+[’']\s+means", re.IGNORECASE)
PAGE_HEADER = "Building Bye-Laws – 2003"


def split_into_clauses(page_text: str) -> list[str]:
    text = page_text.replace(PAGE_HEADER, "").strip()
    starts = [m.start() for m in CLAUSE_START.finditer(text)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    bounds = starts + [len(text)]
    return [text[bounds[i]:bounds[i + 1]].strip() for i in range(len(starts))]


def topics_in(text: str) -> list[str]:
    return [topic for topic, patterns in TOPIC_PATTERNS.items()
            if any(re.search(p, text, re.IGNORECASE) for p in patterns)]


def extract_candidates(pdf_path: Path) -> list[dict]:
    candidates = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for chunk in split_into_clauses(page.extract_text() or ""):
                flat = " ".join(chunk.split())
                if len(flat) < 40 or DEFINITION_PATTERN.search(flat):
                    continue
                topics = topics_in(flat)
                if not topics:
                    continue
                m = CLAUSE_ID.match(flat)
                candidates.append({
                    "page": page_num,
                    "clause_ref": m.group(1) if m else None,  # None = continues from previous page
                    "topics": topics,
                    "text": flat,
                })
    return candidates


if __name__ == "__main__":
    candidates = extract_candidates(PDF_PATH)
    by_topic = {t: sum(t in c["topics"] for c in candidates) for t in TOPIC_PATTERNS}
    output = {
        "source": "Bangalore-Building-Byelaws.pdf (BBMP Building Bye-Laws, 2003)",
        "method": "pdfplumber text, split at numbered clause headings, whole-word topic matching",
        "note": "Candidates only. Persona-facing summaries are curated in data/curated/bylaws_curated.json.",
        "candidate_count": len(candidates),
        "count_by_topic": by_topic,
        "candidates": candidates,
    }
    with open(OUT / "bylaw_candidates.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"{len(candidates)} candidate clauses -> {OUT / 'bylaw_candidates.json'}")
    for topic, n in by_topic.items():
        print(f"  {topic}: {n}")
