"""Agent 2 Age Banding & Dosage Lookup.

Maps numerical age to standardized dataset bands and resolves age-specific
dosage instructions strictly via deterministic dictionary lookups without
LLM arithmetic.
"""

from typing import Any, Literal

DosageOrigin = Literal["from_source", "derived"]


def age_to_band(age: int) -> str:
    """Maps a user age in years to the dataset's standardized age band key.

    Args:
        age: Age in years (1 to 120).

    Returns:
        Age band string key.

    Raises:
        ValueError: If age is outside the valid range [1, 120].
    """
    if not isinstance(age, int) or age < 1 or age > 120:
        raise ValueError(f"Age must be an integer between 1 and 120, got: {age}")

    if age < 3:
        return "infant_under_3"
    elif age < 12:
        return "child_3_12_years"
    elif age < 18:
        return "adolescent_12_17_years"
    elif age <= 60:
        return "adult_18_60_years"
    else:
        return "elderly_60_plus_years"


def lookup_dosage(row: dict[str, Any], band: str) -> tuple[str, DosageOrigin]:
    """Looks up the age-specific dosage from the record's age_suitability table.

    Args:
        row: Dataset record.
        band: Age band key from age_to_band().

    Returns:
        Tuple of (dosage_instruction_string, dosage_origin).
        dosage_origin is 'from_source' if directly found in data, or 'derived'
        if inferred from standard clinical notes.
    """
    if band == "infant_under_3":
        return (
            "Not specified for infants under 3 years; strictly consult an Ayurvedic or paediatric specialist.",
            "from_source",
        )

    age_suit = row.get("age_suitability", {})
    if not isinstance(age_suit, dict):
        return ("Not specified in source, consult a practitioner", "from_source")

    adjustments = age_suit.get("dosage_adjustments_by_age", {})
    if isinstance(adjustments, dict) and band in adjustments and adjustments[band]:
        adj_text = str(adjustments[band]).strip()
        # Append frequency if adult standard
        base_dosage = row.get("dosage", {})
        freq = base_dosage.get("frequency", "") if isinstance(base_dosage, dict) else ""
        if freq and "daily" not in adj_text.lower():
            adj_text = f"{adj_text} ({freq})"
        return (adj_text, "from_source")

    # Fallback for adult if explicitly adult band but adjustments key missing
    if band == "adult_18_60_years":
        base_dosage = row.get("dosage", {})
        if isinstance(base_dosage, dict):
            amt = base_dosage.get("amount", "")
            freq = base_dosage.get("frequency", "")
            dur = base_dosage.get("duration", "")
            combined = f"{amt} {freq}".strip()
            if dur:
                combined += f" for {dur}"
            if combined:
                return (combined, "from_source")

    # Fallback for elderly if notes specify reduction
    if band == "elderly_60_plus_years":
        notes = age_suit.get("notes", "")
        if "reduced" in notes.lower() or "renal" in notes.lower() or "lower end" in notes.lower():
            return (
                "Elderly: Start at lower end of standard dose (reduce by 25-30%); consult practitioner.",
                "derived",
            )

    return ("Not specified in source, consult a practitioner", "from_source")
