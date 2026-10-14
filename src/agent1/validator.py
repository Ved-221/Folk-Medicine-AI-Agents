"""Agent 1 Extractor Validator.

Validates LLM-extracted context blocks against schema rules, strictly verifies
that all evidence quotes appear verbatim in the source record text, computes
an evidence-grounded confidence score, and assigns review status.
"""

import re
from typing import Any

from src.common.config import AUTO_ACCEPT_CONFIDENCE_THRESHOLD
from src.common.schema import ExtractedBlock, ReviewStatus


def normalize_text_for_search(text: str) -> str:
    """Normalizes whitespace and casing for robust substring matching."""
    if not text:
        return ""
    # Collapse whitespace, normalize unicode quotes/dashes
    cleaned = re.sub(r"\s+", " ", text)
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-")
    return cleaned.strip().lower()


def build_record_text(row: dict[str, Any]) -> str:
    """Constructs a clean textual representation of authentic row data for LLM extraction.

    Strictly excludes synthetic scaffolding fields (dosage_model, reasoning_steps, etc.).
    """
    parts: list[str] = []

    # 1. Condition & Category
    cond = row.get("condition", "").replace("_", " ")
    cat = row.get("condition_category", "")
    parts.append(f"Condition: {cond} ({cat})")

    # 2. Symptoms
    symptoms = row.get("symptoms", [])
    if isinstance(symptoms, list) and symptoms:
        parts.append(f"Symptoms: {'; '.join(symptoms)}")

    # 3. Doshas
    doshas = row.get("dosha_involvement", [])
    if isinstance(doshas, list) and doshas:
        parts.append(f"Dosha Involvement: {', '.join(doshas)}")

    # 4. Remedy
    remedy = row.get("remedy", {})
    if isinstance(remedy, dict):
        r_name = remedy.get("name", "")
        r_type = remedy.get("type", "")
        parts.append(f"Remedy: {r_name} (Type: {r_type})")

    # 5. Dosage & Preparation
    dosage = row.get("dosage", {})
    if isinstance(dosage, dict):
        amt = dosage.get("amount", "")
        freq = dosage.get("frequency", "")
        dur = dosage.get("duration", "")
        parts.append(f"Dosage details: amount={amt}, frequency={freq}, duration={dur}")

    # 6. Age suitability notes
    age_notes = row.get("age_suitability", {}).get("notes", "")
    if age_notes:
        parts.append(f"Age considerations: {age_notes}")

    # 7. Gender / Pregnancy notes
    gender = row.get("gender_considerations", {})
    if isinstance(gender, dict):
        p_note = gender.get("pregnancy_note", "")
        b_note = gender.get("breastfeeding_note", "")
        if p_note or b_note:
            parts.append(f"Pregnancy/Lactation notes: {p_note} {b_note}".strip())

    # 8. Contraindications & Escalation
    contra = row.get("contraindications", [])
    if isinstance(contra, list) and contra:
        parts.append(f"Contraindications: {'; '.join(contra)}")

    esc = row.get("escalation", {})
    if isinstance(esc, dict) and esc.get("condition"):
        parts.append(f"Escalation conditions: {esc.get('condition')}")

    # 9. Lifestyle & Dietary
    lifestyle = row.get("lifestyle_factors", {})
    if isinstance(lifestyle, dict):
        diet = lifestyle.get("diet_recommendations", [])
        avoid = lifestyle.get("foods_to_avoid", [])
        if diet:
            parts.append(f"Diet recommendations: {'; '.join(diet)}")
        if avoid:
            parts.append(f"Foods to avoid: {'; '.join(avoid)}")

    return "\n".join(parts)


def validate_extracted_block(
    extracted: dict[str, Any],
    source_text: str,
    threshold: float = AUTO_ACCEPT_CONFIDENCE_THRESHOLD,
) -> tuple[bool, str, ExtractedBlock]:
    """Validates an extracted block and computes evidence-grounded confidence.

    Rules:
    1. Mandatory fields must be present and correctly typed.
    2. Every quote in `evidence` must match a substring in `source_text`.
    3. Calculates confidence based on evidence quote coverage and field completeness.
    4. Routes caution flags (pregnancy/child) to `needs_review`.
    5. Returns (is_valid, validation_message, validated_extracted_block).
    """
    validated = dict(extracted)
    norm_source = normalize_text_for_search(source_text)

    # 1. Type and presence verification
    symptom_terms = validated.get("symptom_terms")
    if not isinstance(symptom_terms, list) or len(symptom_terms) == 0:
        validated["review_status"] = "extraction_failed"
        validated["confidence"] = 0.0
        return False, "Missing or invalid symptom_terms list", validated  # type: ignore

    canonical_ingredients = validated.get("canonical_ingredients")
    if not isinstance(canonical_ingredients, list):
        canonical_ingredients = []
        validated["canonical_ingredients"] = []

    # 2. Verbatim evidence quotes check
    evidence = validated.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
        validated["evidence"] = {}

    valid_quotes = 0
    total_quotes = len(evidence)
    failed_quotes: list[str] = []

    for field_name, quote in evidence.items():
        if not isinstance(quote, str) or not quote.strip():
            continue
        norm_quote = normalize_text_for_search(quote)
        if norm_quote in norm_source:
            valid_quotes += 1
        else:
            failed_quotes.append(f"{field_name}: '{quote}'")

    if total_quotes > 0 and valid_quotes < total_quotes:
        # Quote did not appear in source text -> potential hallucination
        validated["review_status"] = "needs_review"
        quote_fail_msg = f"Evidence quote(s) not found in source text: {'; '.join(failed_quotes)}"
    else:
        quote_fail_msg = ""

    # 3. Confidence calculation
    # Base confidence: proportion of verified quotes (or baseline 0.75 if no quotes required)
    quote_ratio = (valid_quotes / total_quotes) if total_quotes > 0 else 0.5
    completeness_score = 0.0
    if symptom_terms:
        completeness_score += 0.35
    if canonical_ingredients:
        completeness_score += 0.35
    if validated.get("topic_tags"):
        completeness_score += 0.15
    if validated.get("preparation"):
        completeness_score += 0.15

    raw_confidence = (quote_ratio * 0.6) + (completeness_score * 0.4)
    # Penalize if quote validation failed
    if failed_quotes:
        raw_confidence = min(raw_confidence, 0.40)

    confidence = round(max(0.0, min(1.0, raw_confidence)), 2)
    validated["confidence"] = confidence

    # 4. Review status assignment
    has_preg_caution = bool(validated.get("mentions_pregnancy_caution"))
    has_child_caution = bool(validated.get("mentions_child_caution"))

    if failed_quotes:
        validated["review_status"] = "needs_review"
        return False, quote_fail_msg, validated  # type: ignore

    if has_preg_caution or has_child_caution:
        # Caution flags can only ADD caution -> must be reviewed by human
        validated["review_status"] = "needs_review"
        return True, "Flagged for review due to pregnancy/child caution", validated  # type: ignore

    if confidence >= threshold:
        validated["review_status"] = "auto_accepted"
        return True, "Auto accepted", validated  # type: ignore
    else:
        validated["review_status"] = "needs_review"
        return True, f"Confidence {confidence} below threshold {threshold}", validated  # type: ignore
