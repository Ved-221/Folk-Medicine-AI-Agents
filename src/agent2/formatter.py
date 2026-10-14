"""Agent 2 LLM Formatter.

Strictly constrains Gemini 2.5 Flash to rephrase facts present in the retrieved
and trimmed source record into the 6-field output format. Forbids adding external
facts. Provides a deterministic Python template fallback if JSON parsing fails.
"""

import json
from typing import Any

from src.agent1.trimmer import trim_row
from src.common.llm_client import LLMClient, default_client

FORMATTER_PROMPT_TEMPLATE = """You are formatting data from a vetted folk-medicine dataset.
Use ONLY the DATA below. Do not add remedies, doses, ingredients or advice from your own knowledge.
If a field is missing, write "Not specified in source".
Return strictly a JSON object with keys:
- remedy (string)
- dosage_by_age (string)
- ingredients (array of strings)
- lifestyle (string)
- precautions (string)
- foods_to_avoid (array of strings)

USER: age band = {band}, doshas = {doshas}, condition = {condition}
RESOLVED_AGE_DOSAGE: {dosage_by_age}
DATA:
{trimmed_row_json}
"""


def deterministic_fallback_format(
    trimmed_row: dict[str, Any],
    age_band_dosage: str,
    user_input: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic fallback formatting when the LLM is unavailable or fails."""
    remedy_dict = trimmed_row.get("remedy", {})
    remedy_name = remedy_dict.get("name", "Not specified in source") if isinstance(remedy_dict, dict) else str(remedy_dict)
    remedy_type = remedy_dict.get("type", "") if isinstance(remedy_dict, dict) else ""
    full_remedy = f"{remedy_name} ({remedy_type})" if remedy_type else remedy_name

    # Ingredients
    raw_ingredients = trimmed_row.get("ingredients")
    if isinstance(raw_ingredients, list) and raw_ingredients:
        ingredients = [str(i) for i in raw_ingredients]
    elif isinstance(raw_ingredients, str) and raw_ingredients:
        ingredients = [raw_ingredients]
    else:
        ingredients = [remedy_name]

    # Lifestyle
    lifestyle_dict = trimmed_row.get("lifestyle_factors", {})
    diet = []
    foods_avoid = []
    if isinstance(lifestyle_dict, dict):
        diet = lifestyle_dict.get("diet_recommendations", [])
        foods_avoid = lifestyle_dict.get("foods_to_avoid", [])
    lifestyle_str = "; ".join(diet) if diet else "Not specified in source"

    # Precautions
    contra = trimmed_row.get("contraindications", [])
    precautions_str = "; ".join(contra) if contra else "Standard herbal precautions apply."

    return {
        "remedy": full_remedy,
        "dosage_by_age": age_band_dosage,
        "ingredients": ingredients,
        "lifestyle": lifestyle_str,
        "precautions": precautions_str,
        "foods_to_avoid": foods_avoid if foods_avoid else ["Not specified in source"],
    }


def format_recommendation(
    trimmed_row: dict[str, Any],
    age_band_dosage: str,
    age_band: str,
    user_input: dict[str, Any],
    client: LLMClient | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Formats retrieved remedy data into the 6 standard fields via Gemini or fallback.

    Args:
        trimmed_row: Cleaned and trimmed dataset record.
        age_band_dosage: Dosage string resolved by age_band.py.
        age_band: Standardized age band name.
        user_input: User request context.
        client: LLMClient instance.
        use_cache: Whether to use persistent disk cache.

    Returns:
        Dictionary with keys: remedy, dosage_by_age, ingredients, lifestyle, precautions, foods_to_avoid.
    """
    llm = client or default_client

    safe_trimmed = trim_row(trimmed_row)
    trimmed_json = json.dumps(safe_trimmed, ensure_ascii=False, indent=2)

    prompt = FORMATTER_PROMPT_TEMPLATE.format(
        band=age_band,
        doshas=", ".join(user_input.get("doshas", [])),
        condition=user_input.get("condition", ""),
        dosage_by_age=age_band_dosage,
        trimmed_row_json=trimmed_json,
    )

    schema = {
        "type": "object",
        "properties": {
            "remedy": {"type": "string"},
            "dosage_by_age": {"type": "string"},
            "ingredients": {"type": "array", "items": {"type": "string"}},
            "lifestyle": {"type": "string"},
            "precautions": {"type": "string"},
            "foods_to_avoid": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "remedy",
            "dosage_by_age",
            "ingredients",
            "lifestyle",
            "precautions",
            "foods_to_avoid",
        ],
    }

    try:
        parsed = llm.generate_json(
            prompt=prompt,
            schema=schema,
            prompt_version="agent2_format_v1",
            use_cache=use_cache,
        )
        # Ensure all 6 keys are present
        required_keys = ["remedy", "dosage_by_age", "ingredients", "lifestyle", "precautions", "foods_to_avoid"]
        if all(k in parsed for k in required_keys):
            return parsed
        return deterministic_fallback_format(safe_trimmed, age_band_dosage, user_input)
    except Exception:
        # Fallback to deterministic code template on any LLM or parsing failure
        return deterministic_fallback_format(safe_trimmed, age_band_dosage, user_input)
