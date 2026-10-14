"""Unit tests for Agent 1 extractor validator."""

from src.agent1.validator import (
    build_record_text,
    normalize_text_for_search,
    validate_extracted_block,
)


def test_build_record_text_excludes_synthetic():
    row = {
        "condition": "asthma",
        "condition_category": "respiratory",
        "symptoms": ["Wheezing", "Breathlessness"],
        "dosha_involvement": ["vata", "kapha"],
        "remedy": {"name": "Vasaka decoction", "type": "oral"},
        "dosage_model": {"synthetic_table": 123},
        "reasoning_steps": ["step 1", "step 2"],
    }
    text = build_record_text(row)
    assert "Condition: asthma" in text
    assert "Wheezing" in text
    assert "Vasaka decoction" in text
    assert "synthetic_table" not in text
    assert "reasoning_steps" not in text


def test_validate_verbatim_quote_success():
    source_text = "Patient suffers from dry hacking cough in cold weather. Remedy: Tulsi tea."
    extracted = {
        "symptom_terms": ["dry hacking cough"],
        "canonical_ingredients": ["tulsi"],
        "preparation": "tea",
        "topic_tags": ["respiratory"],
        "mentions_pregnancy_caution": False,
        "mentions_child_caution": False,
        "evidence": {
            "symptom_terms": "dry hacking cough",
            "preparation": "Tulsi tea",
        },
    }
    is_valid, msg, validated = validate_extracted_block(extracted, source_text, threshold=0.70)
    assert is_valid is True
    assert validated["review_status"] == "auto_accepted"
    assert validated["confidence"] >= 0.70


def test_fabricated_quote_fails_validation():
    source_text = "Symptoms include mild headache. Take ginger infusion."
    extracted = {
        "symptom_terms": ["severe migraine with aura"],
        "canonical_ingredients": ["ginger"],
        "topic_tags": ["neurological"],
        "mentions_pregnancy_caution": False,
        "mentions_child_caution": False,
        "evidence": {
            # Quote not in source text
            "symptom_terms": "patient suffers from intense throbbing agony with visual flashes",
        },
    }
    is_valid, msg, validated = validate_extracted_block(extracted, source_text)
    assert is_valid is False
    assert "Evidence quote(s) not found" in msg
    assert validated["review_status"] == "needs_review"
    assert validated["confidence"] <= 0.40


def test_caution_flags_force_needs_review():
    source_text = "Avoid during pregnancy due to uterine stimulating properties."
    extracted = {
        "symptom_terms": ["spasm"],
        "canonical_ingredients": ["aloe"],
        "topic_tags": ["digestive"],
        "mentions_pregnancy_caution": True,
        "mentions_child_caution": False,
        "evidence": {
            "caution": "Avoid during pregnancy",
        },
    }
    is_valid, msg, validated = validate_extracted_block(extracted, source_text)
    assert is_valid is True
    # Caution flags can only add caution -> must be reviewed
    assert validated["review_status"] == "needs_review"
    assert "Flagged for review due to pregnancy/child caution" in msg


def test_missing_mandatory_fields_fails():
    source_text = "Some text."
    extracted = {
        "symptom_terms": [],  # Empty
        "canonical_ingredients": [],
    }
    is_valid, msg, validated = validate_extracted_block(extracted, source_text)
    assert is_valid is False
    assert validated["review_status"] == "extraction_failed"
