"""Agent 1 Trimmer.

Reduces raw/cleaned dataset rows to only the authentic clinical and remedy fields
needed for Agent 2 recommendation, strictly eliminating synthetic scaffolding fields
such as dosage_model, dose_optimization, reasoning_steps, and agent_decision_tree.
"""

from typing import Any

EXCLUDED_KEYS = {
    "dosage_model",
    "dose_optimization",
    "agent_decision_tree",
    "reasoning_steps",
    "final_explanation",
    "treatment_plan",
    "variation_simulations",
    "weight_considerations",
}

REQUIRED_KEYS = [
    "row_id",
    "condition",
    "condition_category",
    "knowledge_source",
    "evidence_level",
    "dosha_involvement",
    "symptoms",
    "severity",
    "remedy",
    "ingredients",
    "dosage",
    "age_suitability",
    "lifestyle_factors",
    "contraindications",
    "comorbidity_cautions",
    "gender_considerations",
    "safe_for_home_remedy",
    "care_context",
    "escalation",
]


def trim_row(row: dict[str, Any]) -> dict[str, Any]:
    """Trims a dataset record to include only authentic clinical fields.

    Args:
        row: Normalized row from clean or enriched dataset.

    Returns:
        Trimmed dictionary containing required keys and guaranteed not
        to contain synthetic scaffolding keys.
    """
    trimmed: dict[str, Any] = {}

    for key in REQUIRED_KEYS:
        if key == "ingredients":
            # Extract ingredients if present directly, in remedy, or in extracted block
            val = row.get("ingredients")
            if val is None and isinstance(row.get("remedy"), dict):
                val = row["remedy"].get("ingredients")
            if val is None and isinstance(row.get("extracted"), dict):
                val = row["extracted"].get("canonical_ingredients")
            trimmed["ingredients"] = val
        else:
            trimmed[key] = row.get(key)

    # Preserve the extracted block if already enriched (v2 architecture)
    if "extracted" in row and isinstance(row["extracted"], dict):
        trimmed["extracted"] = row["extracted"]

    return trimmed
