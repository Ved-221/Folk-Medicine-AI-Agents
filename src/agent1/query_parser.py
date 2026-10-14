"""Agent 1 Query Parser.

Converts free-text user messages into a structured query contract using an
LLM parser backed by strict deterministic code validation for doshas, age boundaries,
and life-threatening emergency red-flag symptoms.
"""

from pathlib import Path
import re
from typing import Any

from src.common.config import RED_FLAGS_LIST, ROOT_DIR, VALID_DOSHAS
from src.common.llm_client import LLMClient, default_client
from src.common.schema import ParsedQuery, QUERY_PARSE_JSON_SCHEMA

PROMPTS_DIR = ROOT_DIR / "src" / "agent1" / "prompts"


def load_query_prompt_template(version: str = "query_parse_v1") -> str:
    """Loads a query parse prompt template from src/agent1/prompts/."""
    file_path = PROMPTS_DIR / f"{version}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Query prompt template {version} not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def detect_code_red_flags(query_text: str) -> list[str]:
    """Scans user query text deterministically against known emergency red flags.

    This code gate runs independently of the LLM to guarantee safety.
    """
    detected: list[str] = []
    clean_text = query_text.lower()
    for flag in RED_FLAGS_LIST:
        if flag.lower() in clean_text:
            detected.append(flag)
    return detected


def validate_and_sanitize_parsed_query(
    raw_parsed: dict[str, Any],
    original_text: str,
) -> ParsedQuery:
    """Applies deterministic code-level validation to model-parsed queries.

    Rules:
    - Normalizes and filters doshas strictly to {vata, pitta, kapha}.
    - Verifies age bounds (1 to 120).
    - Merges model-detected and code-detected emergency red flags.
    - Sets is_clarification_needed if age, doshas, or symptom is missing.
    """
    clean_parsed: ParsedQuery = {}

    # 1. Dosha validation
    raw_doshas = raw_parsed.get("doshas", [])
    valid_doshas: list[str] = []
    if isinstance(raw_doshas, list):
        for d in raw_doshas:
            if isinstance(d, str):
                norm = d.strip().lower()
                if norm == "pita":
                    norm = "pitta"
                elif norm == "vatta":
                    norm = "vata"
                if norm in VALID_DOSHAS and norm not in valid_doshas:
                    valid_doshas.append(norm)
    clean_parsed["doshas"] = valid_doshas

    # 2. Symptom phrase
    symptom = str(raw_parsed.get("symptom") or "").strip()
    clean_parsed["symptom"] = symptom

    # 3. Age validation
    raw_age = raw_parsed.get("age")
    valid_age: int | None = None
    if raw_age is not None:
        try:
            parsed_int = int(raw_age)
            if 1 <= parsed_int <= 120:
                valid_age = parsed_int
        except (ValueError, TypeError):
            valid_age = None
    clean_parsed["age"] = valid_age

    # 4. Duration
    duration = raw_parsed.get("duration")
    clean_parsed["duration"] = str(duration).strip() if duration else None

    # 5. Red-flags merge (Code + Model)
    model_flags = raw_parsed.get("red_flags", [])
    if not isinstance(model_flags, list):
        model_flags = []
    code_flags = detect_code_red_flags(original_text)
    all_flags = list(dict.fromkeys(code_flags + [str(f) for f in model_flags if str(f)]))
    clean_parsed["red_flags"] = all_flags

    # 6. Pregnancy / Breastfeeding
    preg = raw_parsed.get("pregnant_or_breastfeeding")
    clean_parsed["pregnant_or_breastfeeding"] = bool(preg) if preg is not None else None

    # 7. Language
    clean_parsed["language_detected"] = str(raw_parsed.get("language_detected") or "English")

    # 8. Missing fields & Clarification gate
    missing: list[str] = []
    if valid_age is None:
        missing.append("age")
    if not valid_doshas:
        missing.append("dosha")
    if not symptom:
        missing.append("symptom")

    clean_parsed["missing_fields"] = missing
    clean_parsed["is_clarification_needed"] = len(missing) > 0 and len(all_flags) == 0

    return clean_parsed


def parse_query(
    text: str,
    client: LLMClient | None = None,
    prompt_version: str = "query_parse_v1",
    use_cache: bool = True,
) -> ParsedQuery:
    """Parses free-text user query into a validated ParsedQuery object.

    Args:
        text: Free-form user input string.
        client: LLMClient instance (defaults to default_client).
        prompt_version: Version of query parse prompt.
        use_cache: Whether to use disk caching.

    Returns:
        Structured ParsedQuery object with code validation applied.
    """
    clean_text = text.strip()
    if not clean_text:
        return {
            "doshas": [],
            "symptom": "",
            "age": None,
            "duration": None,
            "red_flags": [],
            "pregnant_or_breastfeeding": None,
            "language_detected": "Unknown",
            "missing_fields": ["age", "dosha", "symptom"],
            "is_clarification_needed": True,
        }

    # 1. Early deterministic red-flag check
    early_red_flags = detect_code_red_flags(clean_text)

    # 2. Invoke LLM parser
    llm = client or default_client
    prompt_template = load_query_prompt_template(prompt_version)
    prompt = prompt_template.replace("{query_text}", clean_text)

    try:
        raw_parsed = llm.generate_json(
            prompt=prompt,
            schema=QUERY_PARSE_JSON_SCHEMA,
            prompt_version=prompt_version,
            use_cache=use_cache,
        )
    except Exception:
        # Fallback to simple extraction on parse failure
        raw_parsed = {
            "doshas": [d for d in ["vata", "pitta", "kapha"] if d in clean_text.lower()],
            "symptom": clean_text,
            "age": None,
            "duration": None,
            "red_flags": early_red_flags,
            "pregnant_or_breastfeeding": None,
            "language_detected": "English",
            "missing_fields": ["age"],
        }

    # 3. Apply deterministic code validation
    return validate_and_sanitize_parsed_query(raw_parsed, clean_text)
