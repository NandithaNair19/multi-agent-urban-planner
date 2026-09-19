import json
import re
from pathlib import Path


def _parse_markdown_front_matter(path: Path) -> tuple[dict, str]:
    content = path.read_text(encoding="utf-8").strip()
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        raise ValueError(f"Persona file does not contain valid JSON front matter: {path}")

    metadata = json.loads(match.group(1))
    body = match.group(2).strip()
    return metadata, body


def _normalize_persona(persona: dict, source_path: Path) -> dict:
    if not persona.get("id"):
        persona["id"] = source_path.stem

    if not persona.get("label"):
        persona["label"] = persona["id"].replace("-", " ").title()

    persona.setdefault("weight", 1)
    persona.setdefault("priorities", [])
    persona.setdefault("relevant_bylaws", [])
    persona.setdefault("voice_and_tone", "")

    if not persona.get("system_prompt_fragment"):
        persona["system_prompt_fragment"] = ""

    return persona


def load_persona_registry(personas_dir: str | Path) -> list[dict]:
    base_dir = Path(personas_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"No persona directory found at {base_dir}")

    persona_files = sorted(base_dir.glob("*.md"))
    if not persona_files:
        persona_files = sorted(base_dir.glob("*.json"))

    registry = []
    seen_ids = set()

    for path in persona_files:
        if path.suffix.lower() == ".md":
            metadata, body = _parse_markdown_front_matter(path)
            persona = _normalize_persona(metadata, path)
            persona["system_prompt_fragment"] = body or persona.get("system_prompt_fragment", "")
        else:
            with path.open("r", encoding="utf-8") as fh:
                persona = json.load(fh)
            persona = _normalize_persona(persona, path)

        persona_id = persona["id"]
        if persona_id in seen_ids:
            raise ValueError(f"Duplicate persona id found: {persona_id}")
        seen_ids.add(persona_id)
        registry.append(persona)

    return registry


def load_persona_by_id(persona_id: str, personas_dir: str | Path) -> dict:
    for persona in load_persona_registry(personas_dir):
        if persona["id"] == persona_id:
            return persona
    raise FileNotFoundError(f"No persona found with id '{persona_id}' in {personas_dir}")
