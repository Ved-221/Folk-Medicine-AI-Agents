"""Unit tests for Agent 1 extractor and review queue."""

from unittest.mock import MagicMock
from src.agent1.extractor import enrich_dataset, enrich_row
from src.agent1.review_queue import (
    get_pending_queue_items,
    resolve_queue_item,
    sync_rows_to_review_queue,
)
from src.common.llm_client import LLMClient


def test_enrich_row_success(tmp_path):
    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "gemini-2.5-flash"
    mock_client.generate_json.return_value = {
        "symptom_terms": ["hacking cough"],
        "canonical_ingredients": ["tulsi", "honey"],
        "ingredient_aliases": {},
        "preparation": "warm infusion",
        "season_climate_context": "winter",
        "usage_context": "after food",
        "topic_tags": ["respiratory"],
        "mentions_pregnancy_caution": False,
        "mentions_child_caution": False,
        "evidence": {
            "symptom_terms": "hacking cough",
            "canonical_ingredients": "Tulsi with honey",
            "preparation": "warm infusion",
        },
    }

    sample_row = {
        "row_id": "row_101",
        "condition": "cough",
        "symptoms": ["hacking cough", "fever"],
        "dosha_involvement": ["vata", "kapha"],
        "remedy": {"name": "Tulsi with honey", "type": "oral"},
        "dosage": {"amount": "warm infusion"},
    }

    enriched = enrich_row(sample_row, client=mock_client, use_cache=False)

    assert "extracted" in enriched
    assert enriched["condition"] == "cough"  # Original field intact
    assert enriched["row_id"] == "row_101"
    assert enriched["extracted"]["symptom_terms"] == ["hacking cough"]
    assert enriched["extracted"]["canonical_ingredients"] == ["tulsi", "honey"]
    assert enriched["extracted"]["review_status"] == "auto_accepted"
    assert enriched["extracted"]["confidence"] >= 0.70


def test_enrich_row_handles_llm_failure(tmp_path):
    mock_client = MagicMock(spec=LLMClient)
    mock_client.model_name = "gemini-2.5-flash"
    mock_client.generate_json.side_effect = RuntimeError("API Rate limit exhausted")

    sample_row = {"row_id": "row_102", "condition": "fever"}
    enriched = enrich_row(sample_row, client=mock_client, use_cache=False)

    assert "extracted" in enriched
    assert enriched["extracted"]["review_status"] == "extraction_failed"
    assert enriched["extracted"]["confidence"] == 0.0


def test_review_queue_lifecycle(tmp_path):
    queue_file = tmp_path / "test_queue.jsonl"

    rows = [
        {
            "row_id": "row_201",
            "condition": "chronic_ulcer",
            "extracted": {
                "symptom_terms": ["ulcer"],
                "mentions_pregnancy_caution": True,
                "confidence": 0.85,
                "review_status": "needs_review",
            },
        },
        {
            "row_id": "row_202",
            "condition": "mild_cold",
            "extracted": {
                "symptom_terms": ["runny nose"],
                "mentions_pregnancy_caution": False,
                "confidence": 0.90,
                "review_status": "auto_accepted",
            },
        },
    ]

    # Sync to queue
    added = sync_rows_to_review_queue(rows, queue_path=queue_file)
    assert added == 1

    pending = get_pending_queue_items(queue_path=queue_file)
    assert len(pending) == 1
    assert pending[0]["row_id"] == "row_201"
    assert pending[0]["resolved"] is False

    # Resolve item
    success = resolve_queue_item(
        row_id="row_201",
        resolution="reviewed_ok",
        reviewer_notes="Verified by faculty supervisor",
        queue_path=queue_file,
    )
    assert success is True

    pending_after = get_pending_queue_items(queue_path=queue_file)
    assert len(pending_after) == 0
