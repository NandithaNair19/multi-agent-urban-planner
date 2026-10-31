"""
Step 2 of bylaw processing: check the curated bylaw summaries against the
PDF, then write the file the persona agents read.

For every curated clause, each verbatim quote must appear on the stated
page of the PDF (whitespace-normalised). If any quote is missing, the
script stops and writes nothing, so a summary cannot drift away from
its source without someone noticing.

Run:    python3 data/build_bylaws_summary.py
Input:  data/curated/bylaws_curated.json, data/raw/Bangalore-Building-Byelaws.pdf
Output: data/processed/bylaws_summary.json
"""
import json
import sys
from pathlib import Path

import pdfplumber

DATA = Path(__file__).parent
PDF_PATH = DATA / "raw" / "Bangalore-Building-Byelaws.pdf"
CURATED = DATA / "curated" / "bylaws_curated.json"
OUT = DATA / "processed" / "bylaws_summary.json"


def flat(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    with pdfplumber.open(PDF_PATH) as pdf:
        pages = {i: flat(p.extract_text() or "") for i, p in enumerate(pdf.pages, start=1)}

    failures = []
    for clause in curated["clauses"]:
        for ev in clause["evidence"]:
            if flat(ev["quote"]) not in pages.get(ev["page"], ""):
                failures.append(f"{clause['id']}: quote not found on page {ev['page']}: {ev['quote']!r}")
        status = "FAIL" if any(f.startswith(clause["id"] + ":") for f in failures) else "ok"
        print(f"  [{status}] {clause['id']} (pages {sorted({e['page'] for e in clause['evidence']})})")

    if failures:
        print("\nNot written. Fix these first:")
        for f in failures:
            print("  -", f)
        sys.exit(1)

    output = {
        "source": curated["source"],
        "note": curated["note"],
        "clauses": [
            {
                "id": c["id"],
                "topic": c["topic"],
                "pages": sorted({e["page"] for e in c["evidence"]}),
                "summary": c["summary"],
            }
            for c in curated["clauses"]
        ],
        "relevance_for_personas": curated["relevance_for_personas"],
    }
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nAll quotes found. Wrote {OUT}")


if __name__ == "__main__":
    main()
