"""
Orchestrator: runs all 3 personas (poor, middle, wealthy) against the
SAME proposed urban-planning scenario. Since personas are unweighted,
each gets exactly one voice in the output -- no averaging, no scoring.

Run:
    python3 agents/orchestrator.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from persona_agent import PersonaAgent

PERSONA_IDS = ["poor", "middle", "wealthy"]

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_scenario(scenario: str):
    print(f"\nSCENARIO:\n{scenario}\n")
    print("=" * 70)

    results = []
    for pid in PERSONA_IDS:
        agent = PersonaAgent(pid)
        result = agent.react(scenario)
        results.append(result)
        print(f"\n[{result['persona_label']}]")
        print(result["reaction"])
        print("-" * 70)

    output = {
        "timestamp": datetime.now().isoformat(),
        "ward": "Koramangala (Ward 151)",
        "scenario": scenario,
        "personas_unweighted": True,
        "reactions": results,
    }
    out_path = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full run to: {out_path}")
    return output


if __name__ == "__main__":
    demo_scenario = (
        "The BBMP is proposing to convert a vacant lot near the "
        "Sony Signal to Maharaja Signal stretch (a known flood-vulnerable "
        "point) into a flood retention pond with a small public park on top, "
        "as part of stormwater drain upgrades in Koramangala ward."
    )
    run_scenario(demo_scenario)
