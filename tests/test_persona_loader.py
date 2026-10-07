import unittest
from pathlib import Path

from agents.persona_loader import load_persona_registry


BASE_DIR = Path(__file__).resolve().parent.parent


class PersonaLoaderTests(unittest.TestCase):
    def test_persona_registry_loads_markdown_personas(self):
        personas = load_persona_registry(BASE_DIR / "personas")
        persona_ids = {persona["id"] for persona in personas}

        self.assertTrue({"poor", "middle", "wealthy"}.issubset(persona_ids))
        self.assertTrue(all(persona.get("system_prompt_fragment") for persona in personas))
        self.assertTrue(all(persona.get("label") for persona in personas))


if __name__ == "__main__":
    unittest.main()
