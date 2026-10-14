"""Agent 2 Recommender Pipeline.

Orchestrates the end-to-end pipeline:
1. Input validation / query parsing.
2. Deterministic red-flag screening.
3. Grounded candidate retrieval (dosha filter + hybrid/fuzzy matching).
4. Age-band dosage dictionary lookup.
5. Deterministic safety gate evaluation (short-circuits before LLM).
6. Constrained LLM formatting.
7. Provenance attachment and final output delivery.
"""

from pathlib import Path
from typing import Any

from src.agent1.loader import load_dataset
from src.agent1.query import find_hybrid_matches, find_matches
from src.agent1.query_parser import detect_code_red_flags, parse_query
from src.agent1.trimmer import trim_row
from src.agent2.age_band import age_to_band, lookup_dosage
from src.agent2.formatter import format_recommendation
from src.agent2.safety import safety_check
from src.common.config import CLEAN_DATA_PATH, ENRICHED_DATA_PATH
from src.common.llm_client import LLMClient, default_client
from src.common.schema import Agent2Input, Agent2Output, ProvenanceInfo


def get_recommendation(
    doshas: list[str],
    condition: str,
    age: int,
    pregnant: bool = False,
    breastfeeding: bool = False,
    rows: list[dict[str, Any]] | None = None,
    client: LLMClient | None = None,
    use_cache: bool = True,
) -> Agent2Output:
    """Executes the deterministic recommendation pipeline for structured input.

    Args:
        doshas: User dosha list, e.g. ['vata', 'pitta'].
        condition: Symptom or disease query string.
        age: User age in years (1 to 120).
        pregnant: Pregnancy toggle.
        breastfeeding: Breastfeeding toggle.
        rows: Optional pre-loaded dataset rows.
        client: LLMClient instance.
        use_cache: Whether to use disk cache.

    Returns:
        Standardized Agent2Output dictionary.
    """
    llm = client or default_client

    # Step 1: Input contract validation
    if not isinstance(age, int) or age < 1 or age > 120:
        raise ValueError(f"Age must be an integer between 1 and 120, got: {age}")

    clean_doshas = [d.strip().lower() for d in doshas if isinstance(d, str)]
    clean_condition = condition.strip()

    user_context: Agent2Input = {
        "doshas": clean_doshas,
        "condition": clean_condition,
        "age": age,
        "pregnant": pregnant,
        "breastfeeding": breastfeeding,
    }

    # Load dataset if not provided (prefer enriched if available, else clean)
    if rows is None:
        dataset_path = ENRICHED_DATA_PATH if ENRICHED_DATA_PATH.exists() else CLEAN_DATA_PATH
        rows = load_dataset(dataset_path)

    # Step 2: Grounded retrieval (deterministic dosha filter + fuzzy/hybrid scoring)
    matches = find_matches(rows, doshas=clean_doshas, symptom=clean_condition, top_k=1)
    if not matches:
        return {
            "status": "no_match",
            "source_row_id": None,
            "remedy": None,
            "dosage_by_age": None,
            "ingredients": [],
            "lifestyle": None,
            "precautions": "No verified traditional remedy in the dataset matched your symptom and dosha criteria.",
            "foods_to_avoid": [],
            "escalation": "Please consult an Ayurvedic practitioner or general physician for an accurate clinical evaluation.",
            "provenance": {
                "knowledge_source": "classical",
                "evidence_level": "none",
                "dosage_origin": "from_source",
            },
        }

    top_row, match_score = matches[0]
    row_id = str(top_row.get("row_id", "unknown"))

    # Step 3: Trim row to remove synthetic keys
    trimmed = trim_row(top_row)

    # Step 4: Age-band and dosage dictionary lookup (Pure code)
    band = age_to_band(age)
    dosage_text, dosage_origin = lookup_dosage(top_row, band)

    # Step 5: HARD SAFETY GATE (Pure code - runs BEFORE any LLM call)
    safety_result = safety_check(trimmed, user_context)

    # Step 6: Construct provenance metadata
    provenance: ProvenanceInfo = {
        "knowledge_source": top_row.get("knowledge_source", "classical"),
        "evidence_level": str(top_row.get("evidence_level", "classical_text")),
        "dosage_origin": dosage_origin,
    }

    # If Safety Gate triggered: Short-circuit immediately with ZERO LLM calls
    if not safety_result.is_safe:
        return {
            "status": "see_doctor",
            "source_row_id": row_id,
            "remedy": None,
            "dosage_by_age": None,
            "ingredients": [],
            "lifestyle": None,
            "precautions": safety_result.warning_notes or "Immediate medical review required.",
            "foods_to_avoid": [],
            "escalation": safety_result.escalation_reason or "Care context or condition flags indicate self-treatment is unsafe.",
            "provenance": provenance,
        }

    # Step 7: Constrained LLM Formatter (Strict rephrasing only)
    formatted = format_recommendation(
        trimmed_row=trimmed,
        age_band_dosage=dosage_text,
        age_band=band,
        user_input=user_context,
        client=llm,
        use_cache=use_cache,
    )

    return {
        "status": "ok",
        "source_row_id": row_id,
        "remedy": formatted.get("remedy"),
        "dosage_by_age": formatted.get("dosage_by_age"),
        "ingredients": formatted.get("ingredients", []),
        "lifestyle": formatted.get("lifestyle"),
        "precautions": formatted.get("precautions"),
        "foods_to_avoid": formatted.get("foods_to_avoid", []),
        "escalation": safety_result.warning_notes,
        "provenance": provenance,
    }


def get_recommendation_from_text(
    text: str,
    rows: list[dict[str, Any]] | None = None,
    client: LLMClient | None = None,
    use_cache: bool = True,
) -> Agent2Output:
    """Executes end-to-end recommendation from unstructured user free text.

    Args:
        text: Natural language user message.
        rows: Optional dataset rows.
        client: LLMClient instance.
        use_cache: Whether to use disk cache.

    Returns:
        Agent2Output object.
    """
    llm = client or default_client
    clean_text = text.strip()

    # 1. Immediate deterministic red-flag screening (Zero LLM calls for emergency symptoms)
    early_red_flags = detect_code_red_flags(clean_text)
    if early_red_flags:
        red_flags_str = ", ".join(early_red_flags)
        return {
            "status": "see_doctor",
            "source_row_id": None,
            "remedy": None,
            "dosage_by_age": None,
            "ingredients": [],
            "lifestyle": None,
            "precautions": f"EMERGENCY RED FLAGS DETECTED: {red_flags_str}",
            "foods_to_avoid": [],
            "escalation": f"Your symptoms include critical red flags ({red_flags_str}). Stop self-treatment and proceed to emergency hospital care immediately.",
            "provenance": {
                "knowledge_source": "classical",
                "evidence_level": "emergency_gate",
                "dosage_origin": "from_source",
            },
        }

    # 2. Parse free text query
    parsed = parse_query(clean_text, client=llm, use_cache=use_cache)
    if parsed.get("red_flags"):
        red_flags_str = ", ".join(parsed["red_flags"])
        return {
            "status": "see_doctor",
            "source_row_id": None,
            "remedy": None,
            "dosage_by_age": None,
            "ingredients": [],
            "lifestyle": None,
            "precautions": f"EMERGENCY RED FLAGS DETECTED: {red_flags_str}",
            "foods_to_avoid": [],
            "escalation": f"Your symptoms include critical red flags ({red_flags_str}). Stop self-treatment and proceed to emergency hospital care immediately.",
            "provenance": {
                "knowledge_source": "classical",
                "evidence_level": "emergency_gate",
                "dosage_origin": "from_source",
            },
        }

    # 3. Clarification gate
    if parsed.get("is_clarification_needed"):
        missing = ", ".join(parsed.get("missing_fields", []))
        return {
            "status": "no_match",
            "source_row_id": None,
            "remedy": None,
            "dosage_by_age": None,
            "ingredients": [],
            "lifestyle": None,
            "precautions": f"Missing required information: {missing}. Traditional remedies require your age and dosha constitution for safe recommendations.",
            "foods_to_avoid": [],
            "escalation": f"Please provide your {missing} so we can provide a personalized and safe recommendation.",
            "provenance": {
                "knowledge_source": "classical",
                "evidence_level": "clarification_gate",
                "dosage_origin": "from_source",
            },
        }

    # 4. Delegate to structured get_recommendation
    age_val = parsed.get("age") or 30
    preg_val = bool(parsed.get("pregnant_or_breastfeeding"))
    doshas_val = parsed.get("doshas", [])
    symptom_val = parsed.get("symptom", text)

    return get_recommendation(
        doshas=doshas_val,
        condition=symptom_val,
        age=age_val,
        pregnant=preg_val,
        rows=rows,
        client=llm,
        use_cache=use_cache,
    )
