"""Unit tests for Agent 1 loader and cleaner."""

import hashlib
from pathlib import Path
from src.agent1.loader import clean_row, load_dataset, process_and_save_dataset
from src.common.config import CLEAN_DATA_PATH, RAW_DATA_PATH, VALID_DOSHAS


def test_clean_row_dosha_normalization():
    raw_sample = {
        "condition": "  Chronic Migraine  ",
        "dosha_involvement": ["Vata", "Pitta", "vata", "rakta", "UNKNOWN_DOSHA"],
        "remedy": {"name": "Pathyadi Kwath", "type": "oral"},
    }
    cleaned = clean_row(raw_sample, default_id="row_001")

    assert cleaned["row_id"] == "row_001"
    assert cleaned["condition"] == "Chronic Migraine"
    # Should lowercase, map rakta to pitta, deduplicate, filter unknown
    assert cleaned["dosha_involvement"] == ["vata", "pitta"]
    assert cleaned["remedy"] == {"name": "Pathyadi Kwath", "type": "oral"}


def test_clean_row_preserves_nested_structure():
    raw_sample = {
        "condition": "fever",
        "dosha_involvement": ["pitta"],
        "age_suitability": {
            "minimum_age": "3 years",
            "dosage_adjustments_by_age": {
                "adult_18_60_years": "standard dose",
                "child_3_12_years": "half dose",
            },
        },
    }
    cleaned = clean_row(raw_sample)
    assert cleaned["age_suitability"]["dosage_adjustments_by_age"]["child_3_12_years"] == "half dose"


def test_process_dataset_end_to_end(tmp_path):
    # Compute SHA256 of raw dataset before processing
    with open(RAW_DATA_PATH, "rb") as f:
        raw_hash_before = hashlib.sha256(f.read()).hexdigest()

    test_clean_output = tmp_path / "cleaned_dataset.jsonl"
    cleaned_rows = process_and_save_dataset(RAW_DATA_PATH, test_clean_output)

    # 1. Total rows count check
    assert len(cleaned_rows) == 500

    # 2. Raw dataset must remain untouched
    with open(RAW_DATA_PATH, "rb") as f:
        raw_hash_after = hashlib.sha256(f.read()).hexdigest()
    assert raw_hash_before == raw_hash_after

    # 3. Dosha values check across all 500 rows
    all_doshas = set()
    for row in cleaned_rows:
        assert "row_id" in row
        assert row["condition"] == row["condition"].strip()
        for d in row["dosha_involvement"]:
            all_doshas.add(d)

    assert all_doshas == VALID_DOSHAS
    assert test_clean_output.exists()
