"""
Parse the BBMP Building Byelaws PDF programmatically

We search the extracted text for paragraphs containing key terms
(slum, EWS/LIG, setback, FAR, land use) and save each matching
paragraph with its page number, so every clause can be traced back
to an exact page in the source PDF.

Run:  python3 data/parse_bylaws.py
Output: data/processed/bylaws_summary.json
"""
import json
import re
from pathlib import Path
import pdfplumber

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "processed"
OUT.mkdir(exist_ok=True)

PDF_PATH = RAW / "Bangalore-Building-Byelaws.pdf"

# Keyword -> topic label, used to tag which paragraphs are relevant
KEYWORD_TOPICS = {
    "slum": "slum_clearance",
    "EWS": "ews_lig_housing",
    "LIG": "ews_lig_housing",
    "setback": "setbacks",
    "FAR": "floor_area_ratio",
    "floor area ratio": "floor_area_ratio",
    "land use": "land_use_change",
}

PERSONA_RELEVANCE = {
    "slum_clearance": "poor",
    "ews_lig_housing": "poor",
    "setbacks": "middle",
    "floor_area_ratio": "middle",
    "land_use_change": "wealthy",
}


# Definition-style paragraphs look like: 'Term' means ... -- these are
# glossary noise, not substantive regulatory clauses, so we skip them.
DEFINITION_PATTERN = re.compile(r"[\u2018']\s*[\w\s\-]+[\u2019']\s+means", re.IGNORECASE)


def extract_matching_paragraphs(pdf_path):
    results = []
    seen_texts = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Split into paragraphs on double newlines / long single newlines
            paragraphs = re.split(r"\n\s*\n", text)
            for para in paragraphs:
                para_clean = para.strip().replace("\n", " ")
                if len(para_clean) < 40:
                    continue  # skip headers/fragments too short to be meaningful
                if DEFINITION_PATTERN.search(para_clean):
                    continue  # skip glossary/definition entries
                for keyword, topic in KEYWORD_TOPICS.items():
                    if keyword.lower() in para_clean.lower():
                        key = (topic, para_clean[:80])  # dedupe by topic+start
                        if key in seen_texts:
                            continue
                        seen_texts.add(key)
                        results.append({
                            "topic": topic,
                            "page": page_num,
                            "text": para_clean[:600],  # cap length per clause
                            "relevant_persona": PERSONA_RELEVANCE.get(topic, "all"),
                        })
                        break  # one topic tag per paragraph 
    return results


clauses = extract_matching_paragraphs(PDF_PATH)

output = {
    "source": "Bangalore-Building-Byelaws.pdf (BBMP Building Bye-Laws, 2003)",
    "extraction_method": "pdfplumber, keyword-matched paragraphs (slum, EWS/LIG, setback, FAR, land use)",
    "note": "City-wide bylaws, not Koramangala-specific. Each clause traceable to its PDF page number.",
    "clause_count": len(clauses),
    "clauses": clauses,
}

with open(OUT / "bylaws_summary.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Extracted {len(clauses)} relevant clauses -> {OUT / 'bylaws_summary.json'}")
for c in clauses[:5]:
    print(f"\n[Page {c['page']}] ({c['topic']}, relevant to: {c['relevant_persona']})")
    print(c["text"][:200], "...")