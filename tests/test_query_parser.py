"""Unit tests for Agent 1 query parser with mocked LLM and code validation."""

from unittest.mock import MagicMock
from src.agent1.query_parser import (
    detect_code_red_flags,
    parse_query,
    validate_and_sanitize_parsed_query,
)
from src.common.llm_client import LLMClient


def test_detect_code_red_flags():
    assert "coughing blood" in detect_code_red_flags("Patient is coughing blood and dizzy.")
    assert "crushing chest pain" in detect_code_red_flags("Sudden crushing chest pain radiating to arm.")
    assert detect_code_red_flags("Just a mild scratchy throat and runny nose.") == []


def test_parse_query_standard():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.generate_json.return_value = {
        "doshas": ["vata", "pitta"],
        "symptom": "dry cough",
        "age": 32,
        "duration": "4 days",
        "red_flags": [],
        "pregnant_or_breastfeeding": False,
        "language_detected": "English",
        "missing_fields": [],
    }

    res = parse_query(
        "I am 32 with vata pitta dosha, having dry cough for 4 days.",
        client=mock_client,
        use_cache=False,
    )
    assert res["doshas"] == ["vata", "pitta"]
    assert res["symptom"] == "dry cough"
    assert res["age"] == 32
    assert res["is_clarification_needed"] is False
    assert res["red_flags"] == []


def test_parse_query_missing_age_triggers_clarification():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.generate_json.return_value = {
        "doshas": ["pitta"],
        "symptom": "acid reflux",
        "age": None,
        "duration": None,
        "red_flags": [],
        "pregnant_or_breastfeeding": None,
        "language_detected": "English",
        "missing_fields": ["age"],
    }

    res = parse_query(
        "Suffering from acid reflux, I have pitta constitution.",
        client=mock_client,
        use_cache=False,
    )
    assert res["age"] is None
    assert "age" in res["missing_fields"]
    assert res["is_clarification_needed"] is True


def test_parse_query_code_red_flag_overrides():
    mock_client = MagicMock(spec=LLMClient)
    # Simulate LLM failing to recognize red flag
    mock_client.generate_json.return_value = {
        "doshas": ["vata"],
        "symptom": "cough",
        "age": 45,
        "duration": None,
        "red_flags": [],
        "pregnant_or_breastfeeding": False,
        "language_detected": "English",
        "missing_fields": [],
    }

    text = "Age 45, vata dosha, coughing up blood since morning."
    res = parse_query(text, client=mock_client, use_cache=False)

    # Code red flag detector must catch it regardless
    assert len(res["red_flags"]) > 0
    assert "coughing up blood" in res["red_flags"]


def test_validate_and_sanitize_dosha_normalization():
    raw = {
        "doshas": ["Vatta", "PITA", "unknown_dosha"],
        "symptom": "headache",
        "age": 200,  # Invalid age
        "red_flags": [],
    }
    clean = validate_and_sanitize_parsed_query(raw, "headache query")
    assert clean["doshas"] == ["vata", "pitta"]
    assert clean["age"] is None
    assert "age" in clean["missing_fields"]
    assert clean["is_clarification_needed"] is True
