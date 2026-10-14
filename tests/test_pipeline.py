"""End-to-end integration tests for Agent 2 recommendation pipeline."""

from unittest.mock import MagicMock
from src.agent1.loader import load_dataset
from src.agent2.pipeline import (
    get_recommendation,
    get_recommendation_from_text,
)
from src.common.config import CLEAN_DATA_PATH
from src.common.llm_client import LLMClient


def get_mock_client():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "gemini-2.5-flash"
    mock_client.generate_json.return_value = {
        "remedy": "Vasaka decoction",
        "dosage_by_age": "standard dose (twice daily)",
        "ingredients": ["Vasaka (Adhatoda vasica)"],
        "lifestyle": "Stay warm, avoid cold breezes.",
        "precautions": "Consult doctor if fever persists.",
        "foods_to_avoid": ["Cold beverages", "Ice cream"],
    }
    return mock_client


def test_pipeline_normal_safe_recommendation():
    rows = load_dataset(CLEAN_DATA_PATH)
    mock_client = get_mock_client()

    rec = get_recommendation(
        doshas=["vata", "pitta"],
        condition="dry cough",
        age=30,
        rows=rows,
        client=mock_client,
        use_cache=False,
    )

    assert rec["status"] == "ok"
    assert rec["source_row_id"] is not None
    assert rec["remedy"] is not None
    assert rec["dosage_by_age"] is not None
    assert len(rec["ingredients"]) > 0
    assert rec["provenance"]["knowledge_source"] in ("classical", "folk")


def test_pipeline_unsafe_row_short_circuits_before_llm():
    rows = load_dataset(CLEAN_DATA_PATH)
    mock_client = get_mock_client()

    # Query matching rheumatoid_arthritis (row_001), which is safe_for_home_remedy=False
    rec = get_recommendation(
        doshas=["vata", "pitta", "kapha"],
        condition="severe joint pain and swelling",
        age=45,
        rows=rows,
        client=mock_client,
        use_cache=False,
    )

    assert rec["status"] == "see_doctor"
    assert rec["remedy"] is None
    assert rec["escalation"] is not None
    # Crucial: LLM formatter was NEVER called!
    mock_client.generate_json.assert_not_called()


def test_pipeline_no_match():
    rows = load_dataset(CLEAN_DATA_PATH)
    mock_client = get_mock_client()

    rec = get_recommendation(
        doshas=["vata"],
        condition="quantum glitch cybernetic anomaly",
        age=25,
        rows=rows,
        client=mock_client,
        use_cache=False,
    )

    assert rec["status"] == "no_match"
    assert rec["remedy"] is None
    mock_client.generate_json.assert_not_called()


def test_pipeline_elderly_dosage():
    rows = load_dataset(CLEAN_DATA_PATH)
    mock_client = get_mock_client()

    rec = get_recommendation(
        doshas=["pitta"],
        condition="indigestion",
        age=75,  # Elderly
        rows=rows,
        client=mock_client,
        use_cache=False,
    )

    assert rec["status"] in ("ok", "see_doctor")
    assert rec["provenance"]["dosage_origin"] in ("from_source", "derived")


def test_pipeline_from_text_emergency_red_flags():
    mock_client = get_mock_client()

    rec = get_recommendation_from_text(
        text="I am 40 years old, vata pitta, coughing up blood and feeling sharp chest pain.",
        client=mock_client,
        use_cache=False,
    )

    assert rec["status"] == "see_doctor"
    assert "EMERGENCY RED FLAGS" in rec["precautions"]
    assert "blood" in rec["escalation"].lower()
    # LLM formatter must never be invoked for medical emergencies
    mock_client.generate_json.assert_not_called()


def test_pipeline_from_text_missing_fields_clarification():
    mock_client = get_mock_client()

    # Query missing age and dosha
    rec = get_recommendation_from_text(
        text="I have a scratchy dry cough.",
        client=mock_client,
        use_cache=False,
    )

    assert rec["status"] == "no_match"
    assert "Missing required information" in rec["precautions"]
