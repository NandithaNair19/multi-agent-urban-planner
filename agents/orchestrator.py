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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from agents.persona_loader import load_persona_registry
except ImportError:
    from persona_loader import load_persona_registry

from persona_agent import PersonaAgent

PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"


def get_persona_ids(persona_ids=None):
    registry = load_persona_registry(PERSONAS_DIR)
    available = [persona["id"] for persona in registry]
    preferred_order = ["poor", "middle", "wealthy"]

    ordered_available = [persona_id for persona_id in preferred_order if persona_id in available]
    ordered_available.extend([persona_id for persona_id in available if persona_id not in preferred_order])

    if persona_ids is None:
        return ordered_available

    selected = []
    missing = []
    for persona_id in persona_ids:
        if persona_id in available:
            selected.append(persona_id)
        else:
            missing.append(persona_id)

    if missing:
        raise ValueError(f"Unknown persona ids requested: {missing}")
    return selected

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_scenario(scenario: str, persona_ids=None):
    active_personas = get_persona_ids(persona_ids)
    print(f"\nSCENARIO:\n{scenario}\n")
    print(f"PERSONAS: {', '.join(active_personas)}")
    print("=" * 70)

    results = []
    for pid in active_personas:
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
