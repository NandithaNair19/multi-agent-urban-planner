"""
PersonaAgent: wraps one persona (poor/middle/wealthy) so it can react
to a proposed urban-planning scenario, grounded in real ward data.

This version uses Google's free Gemini API instead of Claude.

Requires: pip install google-genai
Requires: an environment variable GEMINI_API_KEY set with your free API key
(get one at https://aistudio.google.com/apikey -- no credit card needed).
"""
import os
import json
import sys
from dotenv import load_dotenv
from pathlib import Path
from google import genai

try:
    from agents.persona_loader import load_persona_by_id
except ImportError:
    from persona_loader import load_persona_by_id

load_dotenv()

BASE = Path(__file__).parent.parent
PERSONAS_DIR = BASE / "personas"
WARD_CONTEXT_PATH = BASE / "data" / "processed" / "ward_context.json"
BYLAWS_PATH = BASE / "data" / "processed" / "bylaws_summary.json"

MODEL_NAME = "gemini-3.6-flash"


def load_json(path):
    with open(path) as f:
        return json.load(f)


class PersonaAgent:
    def __init__(self, persona_id: str, client: genai.Client = None, personas_dir: str | Path = PERSONAS_DIR):
        self.personas_dir = Path(personas_dir)
        self.persona = load_persona_by_id(persona_id, self.personas_dir)
        self.ward_context = load_json(WARD_CONTEXT_PATH)
        self.bylaws = load_json(BYLAWS_PATH)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key and client is None:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        self.client = client or genai.Client(api_key=api_key)

    def _build_prompt(self, scenario: str) -> str:
        relevant_clause_ids = self.persona.get("relevant_bylaws", [])
        relevant_clauses = [
            c for c in self.bylaws["clauses"] if c["id"] in relevant_clause_ids
        ]

        prompt = f"""{self.persona['system_prompt_fragment']}

WARD CONTEXT (Koramangala, Ward 151, Bangalore):
- Population: {self.ward_context['census']['population_total']} ({self.ward_context['census']['sc_st_share_pct']}% SC/ST)
- Property tax collections grew {self.ward_context['property_tax'].get('collection_growth_pct', 'N/A')}% over FY2016-17 to FY2018-19
- Known flood-vulnerable point: {self.ward_context['flood_risk']['flood_vulnerable_points'][0]['name']}
- Known flood-prone informal settlement: {self.ward_context['flood_risk']['flood_prone_points'][0]['name']}

RELEVANT BUILDING BYELAWS FOR YOU:
{json.dumps(relevant_clauses, indent=2) if relevant_clauses else "None specifically flagged."}

PROPOSED URBAN PLANNING SCENARIO:
{scenario}

Respond in 2-4 sentences, in character, giving your honest reaction to this
proposal from your own household's perspective. Be specific about how it
would affect you, not generic."""
        return prompt

    def react(self, scenario: str) -> dict:
        prompt = self._build_prompt(scenario)
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return {
            "persona_id": self.persona["id"],
            "persona_label": self.persona["label"],
            "reaction": response.text,
        }
