"""
Step 1: Consolidate all Koramangala (Ward 151) source data into one
ward_context.json. This is the single source of truth every persona
agent reads from downstream.

This version uses only Python's built-in csv/json modules -- no
pandas/numpy needed, to avoid Mac architecture install issues.

Run:  python3 data/build_ward_context.py
Output: data/processed/ward_context.json
"""
import csv
import json
from pathlib import Path

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "processed"
OUT.mkdir(exist_ok=True)

WARD_NO = 151
WARD_NAME = "Koramangala"


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# --- Census (ward-level aggregate) ---
census_rows = read_csv_rows(RAW / "Bangalore-Ward-Census-Data.csv")
row = next(r for r in census_rows if int(r["Ward Num"]) == WARD_NO)
census_block = {
    "population_total": int(row["Population"]),
    "male": int(row["Male"]),
    "female": int(row["Female"]),
    "sc_population": int(row["SC Population"]),
    "st_population": int(row["ST Population"]),
    "assembly_constituency": row["Assembly constituency"],
    "sc_st_share_pct": round(
        100 * (int(row["SC Population"]) + int(row["ST Population"])) / int(row["Population"]), 2
    ),
}

# --- Property tax (ward-level aggregate, 3 financial years) ---
tax_rows_all = read_csv_rows(RAW / "Bangalore-Ward-Property-Tax-Data.csv")
tax_rows = [r for r in tax_rows_all if int(r["Ward Number"]) == WARD_NO]
tax_rows.sort(key=lambda r: r["Finacial Year"])

tax_block = {
    "note": "Ward-level aggregate only — no per-property records, no tax bands.",
    "by_year": [
        {
            "year": r["Finacial Year"],
            "applications": int(r["No of Applications"]),
            "collection_lakhs": float(r["Total Collection (Amount in lakhs)"]),
        }
        for r in tax_rows
    ],
}
if len(tax_block["by_year"]) >= 2:
    first, last = tax_block["by_year"][0], tax_block["by_year"][-1]
    tax_block["collection_growth_pct"] = round(
        100 * (last["collection_lakhs"] - first["collection_lakhs"]) / first["collection_lakhs"], 2
    )

# --- Flood risk (already spatially validated against ward boundary) ---
with open(RAW / "koramangala_flood_risk.json") as f:
    flood_block = json.load(f)

# --- Ward boundary summary (avoid dumping full geometry into every prompt) ---
with open(RAW / "koramangala_ward_boundary.geojson") as f:
    boundary = json.load(f)
props = boundary["features"][0]["properties"]
boundary_block = {
    "ward_no": int(props["WARD_NO"]),
    "ward_name": props["WARD_NAME"],
    "area_sq_km": props["AREA_SQ_KM"],
    "reservation": props.get("RESERVATIO", props.get("RESERVATION")),
    "centroid_lat": props["LAT"],
    "centroid_lon": props["LON"],
}

# --- Assemble ---
ward_context = {
    "ward_no": WARD_NO,
    "ward_name": WARD_NAME,
    "boundary": boundary_block,
    "census": census_block,
    "property_tax": tax_block,
    "flood_risk": flood_block,
}

with open(OUT / "ward_context.json", "w") as f:
    json.dump(ward_context, f, indent=2)

print("Saved:", OUT / "ward_context.json")
print(json.dumps(ward_context, indent=2))
