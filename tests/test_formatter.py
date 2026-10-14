"""Unit tests for Agent 2 formatter with mocked LLM and template fallback."""

from unittest.mock import MagicMock
from src.agent2.formatter import (
    deterministic_fallback_format,
    format_recommendation,
)
from src.common.llm_client import LLMClient


def test_deterministic_fallback_format():
    row = {
        "remedy": {"name": "Ginger decoction", "type": "oral"},
        "ingredients": ["ginger", "black pepper"],
        "lifestyle_factors": {
            "diet_recommendations": ["Eat warm foods", "Drink hot water"],
            "foods_to_avoid": ["Cold beverages", "Ice cream"],
        },
        "contraindications": ["Avoid in active peptic ulcers"],
    }
    fallback = deterministic_fallback_format(
        trimmed_row=row,
        age_band_dosage="250mg twice daily",
        user_input={"doshas": ["vata"], "condition": "cough"},
    )
    assert fallback["remedy"] == "Ginger decoction (oral)"
    assert fallback["dosage_by_age"] == "250mg twice daily"
    assert fallback["ingredients"] == ["ginger", "black pepper"]
    assert "Eat warm foods" in fallback["lifestyle"]
    assert "Cold beverages" in fallback["foods_to_avoid"]
    assert "peptic ulcers" in fallback["precautions"]


def test_format_recommendation_mocked_llm_success():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.generate_json.return_value = {
        "remedy": "Vasaka Syrup",
        "dosage_by_age": "5ml thrice daily",
        "ingredients": ["Adhatoda vasica"],
        "lifestyle": "Stay warm and avoid cold breezes.",
        "precautions": "Consult doctor if cough worsens after 3 days.",
        "foods_to_avoid": ["Sour curd", "Cold water"],
    }

    row = {
        "row_id": "r10",
        "remedy": {"name": "Vasaka Syrup", "type": "oral"},
        "dosage": {"amount": "5ml"},
    }

    formatted = format_recommendation(
        trimmed_row=row,
        age_band_dosage="5ml thrice daily",
        age_band="adult_18_60_years",
        user_input={"doshas": ["vata", "kapha"], "condition": "cough"},
        client=mock_client,
        use_cache=False,
    )

    assert formatted["remedy"] == "Vasaka Syrup"
    assert formatted["dosage_by_age"] == "5ml thrice daily"
    assert formatted["ingredients"] == ["Adhatoda vasica"]
    assert len(formatted["foods_to_avoid"]) == 2


def test_format_recommendation_fallback_on_llm_error():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.generate_json.side_effect = RuntimeError("LLM network timeout")

    row = {
        "row_id": "r11",
        "remedy": {"name": "Sitopaladi Churna", "type": "oral"},
        "dosage": {"amount": "1g"},
        "contraindications": ["Do not take unsupervised during pregnancy."],
    }

    formatted = format_recommendation(
        trimmed_row=row,
        age_band_dosage="1g twice daily",
        age_band="adult_18_60_years",
        user_input={"doshas": ["vata"], "condition": "dry cough"},
        client=mock_client,
        use_cache=False,
    )

    # Must fall back gracefully to deterministic template
    assert "Sitopaladi Churna" in formatted["remedy"]
    assert formatted["dosage_by_age"] == "1g twice daily"
    assert "pregnancy" in formatted["precautions"].lower()
