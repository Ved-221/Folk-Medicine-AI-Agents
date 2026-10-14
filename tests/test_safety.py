"""Unit tests for Agent 2 hard safety gate (test-first development)."""

from src.agent2.safety import SafetyResult, safety_check
from src.common.schema import Agent2Input


def test_safety_check_safe_row():
    safe_row = {
        "row_id": "row_095",
        "condition": "common_cold",
        "safe_for_home_remedy": True,
        "care_context": "home",
        "gender_considerations": {
            "pregnancy_safe": True,
            "pregnancy_note": "Generally safe in moderate culinary doses.",
            "breastfeeding_safe": True,
        },
        "escalation": {"needed": False, "condition": ""},
    }
    user_context: Agent2Input = {
        "doshas": ["vata"],
        "condition": "common cold",
        "age": 30,
        "pregnant": False,
        "breastfeeding": False,
    }

    result = safety_check(safe_row, user_context)
    assert result.is_safe is True
    assert result.status == "ok"
    assert result.escalation_reason is None


def test_safety_check_unsafe_for_home_remedy():
    unsafe_row = {
        "row_id": "row_001",
        "condition": "rheumatoid_arthritis",
        "safe_for_home_remedy": False,
        "care_context": "clinic",
        "escalation": {
            "needed": True,
            "condition": "Seek clinician review if symptoms progress or there is significant joint swelling.",
        },
    }
    user_context: Agent2Input = {
        "doshas": ["vata", "pitta"],
        "condition": "rheumatoid arthritis",
        "age": 45,
        "pregnant": False,
        "breastfeeding": False,
    }

    result = safety_check(unsafe_row, user_context)
    assert result.is_safe is False
    assert result.status == "see_doctor"
    assert "Seek clinician review" in result.escalation_reason


def test_safety_check_emergency_care_context():
    emergency_row = {
        "row_id": "row_002",
        "condition": "bronchial_asthma",
        "safe_for_home_remedy": False,
        "care_context": "emergency",
        "escalation": {
            "needed": True,
            "condition": "Seek urgent care for shortness of breath at rest, chest indrawing, or blue lips.",
        },
    }
    user_context: Agent2Input = {
        "doshas": ["vata"],
        "condition": "asthma",
        "age": 25,
        "pregnant": False,
        "breastfeeding": False,
    }

    result = safety_check(emergency_row, user_context)
    assert result.is_safe is False
    assert result.status == "see_doctor"
    assert "urgent care" in result.escalation_reason.lower()


def test_safety_check_pregnancy_contraindication():
    preg_contra_row = {
        "row_id": "row_050",
        "condition": "digestive_colic",
        "safe_for_home_remedy": True,
        "care_context": "home",
        "gender_considerations": {
            "pregnancy_safe": False,
            "pregnancy_note": "Contains herbs contraindicated during pregnancy; stimulates uterine contractions.",
            "breastfeeding_safe": True,
        },
        "escalation": {"needed": False, "condition": ""},
    }
    user_context: Agent2Input = {
        "doshas": ["pitta"],
        "condition": "digestive colic",
        "age": 28,
        "pregnant": True,  # User is pregnant
        "breastfeeding": False,
    }

    result = safety_check(preg_contra_row, user_context)
    assert result.is_safe is False
    assert result.status == "see_doctor"
    assert "pregnancy" in result.escalation_reason.lower()


def test_safety_check_breastfeeding_contraindication():
    bf_contra_row = {
        "row_id": "row_051",
        "condition": "skin_infection",
        "safe_for_home_remedy": True,
        "care_context": "home",
        "gender_considerations": {
            "pregnancy_safe": True,
            "breastfeeding_safe": False,
            "breastfeeding_note": "Excreted in breast milk; avoid during lactation.",
        },
        "escalation": {"needed": False, "condition": ""},
    }
    user_context: Agent2Input = {
        "doshas": ["pitta"],
        "condition": "skin infection",
        "age": 29,
        "pregnant": False,
        "breastfeeding": True,  # User is breastfeeding
    }

    result = safety_check(bf_contra_row, user_context)
    assert result.is_safe is False
    assert result.status == "see_doctor"
    assert "breastfeeding" in result.escalation_reason.lower() or "lactation" in result.escalation_reason.lower()


def test_safety_check_pending_review_row_adds_warning():
    unreviewed_row = {
        "row_id": "row_080",
        "condition": "mild_rash",
        "safe_for_home_remedy": True,
        "care_context": "home",
        "extracted": {
            "review_status": "needs_review",
            "mentions_pregnancy_caution": True,
        },
        "escalation": {"needed": False, "condition": ""},
    }
    user_context: Agent2Input = {
        "doshas": ["pitta"],
        "condition": "mild rash",
        "age": 30,
        "pregnant": False,
        "breastfeeding": False,
    }

    result = safety_check(unreviewed_row, user_context)
    # Row has extracted caution flag and needs review -> safety gate adds cautionary escalation
    assert "review" in (result.warning_notes or "").lower() or result.status == "see_doctor"
