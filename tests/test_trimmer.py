"""Unit tests for Agent 1 trimmer."""

from src.agent1.loader import load_dataset
from src.agent1.trimmer import EXCLUDED_KEYS, REQUIRED_KEYS, trim_row
from src.common.config import CLEAN_DATA_PATH


def test_trim_row_synthetic_keys_excluded():
    sample_raw = {
        "row_id": "row_001",
        "condition": "cough",
        "dosha_involvement": ["vata"],
        "remedy": {"name": "Sitopaladi", "type": "oral"},
        "dosage_model": {"version": "2.0", "adjustment_tables": {}},
        "treatment_plan": {"agent_decision_tree": {}},
        "reasoning_steps": ["step 1", "step 2"],
        "final_explanation": "summary",
        "weight_considerations": {},
    }
    trimmed = trim_row(sample_raw)

    for ex_key in EXCLUDED_KEYS:
        assert ex_key not in trimmed, f"Synthetic key '{ex_key}' should not be in trimmed output"

    assert trimmed["row_id"] == "row_001"
    assert trimmed["condition"] == "cough"
    assert trimmed["remedy"]["name"] == "Sitopaladi"


def test_trim_row_required_keys_present():
    minimal_row = {"row_id": "row_002", "condition": "fever"}
    trimmed = trim_row(minimal_row)

    for req_key in REQUIRED_KEYS:
        assert req_key in trimmed, f"Required key '{req_key}' missing from trimmed row"

    assert trimmed["dosage"] is None
    assert trimmed["safe_for_home_remedy"] is None


def test_trim_dataset_all_500_rows():
    rows = load_dataset(CLEAN_DATA_PATH)
    assert len(rows) == 500

    for r in rows:
        trimmed = trim_row(r)
        for ex_key in EXCLUDED_KEYS:
            assert ex_key not in trimmed, f"Row {r.get('row_id')} retained synthetic key '{ex_key}'"
        for req_key in REQUIRED_KEYS:
            assert req_key in trimmed
