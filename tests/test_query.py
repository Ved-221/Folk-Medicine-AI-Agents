"""Unit tests for Agent 1 query retrieval layer."""

from src.agent1.loader import load_dataset
from src.agent1.query import find_matches, score_row_fuzzy
from src.common.config import CLEAN_DATA_PATH


def test_score_row_fuzzy():
    row = {
        "condition": "dry_cough",
        "symptoms": ["Tickling in throat", "Hacking non-productive cough"],
    }
    score_exact = score_row_fuzzy("dry cough", row)
    assert score_exact >= 90.0

    score_symptom = score_row_fuzzy("tickling throat", row)
    assert score_symptom >= 70.0

    score_irrelevant = score_row_fuzzy("broken bone fracture", row)
    assert score_irrelevant < 50.0


def test_find_matches_dry_cough():
    rows = load_dataset(CLEAN_DATA_PATH)
    matches = find_matches(rows, doshas=["vata", "pitta"], symptom="dry cough", top_k=3)

    assert len(matches) > 0
    top_row, top_score = matches[0]
    assert top_score >= 50.0
    # Must have vata or pitta involvement
    row_doshas = set(top_row.get("dosha_involvement", []))
    assert bool(row_doshas & {"vata", "pitta"})


def test_find_matches_nonsense_symptom():
    rows = load_dataset(CLEAN_DATA_PATH)
    matches = find_matches(rows, doshas=["vata"], symptom="quantum teleportation cybernetic glitch", top_k=3)
    assert matches == []


def test_find_matches_tridoshic():
    rows = load_dataset(CLEAN_DATA_PATH)
    matches = find_matches(rows, doshas=["vata", "pitta", "kapha"], symptom="joint pain", top_k=5)
    assert len(matches) > 0

    # Ensure results are sorted descending
    scores = [score for _, score in matches]
    assert scores == sorted(scores, reverse=True)


def test_dosha_filtering_exclusion():
    mock_rows = [
        {
            "row_id": "r1",
            "condition": "dry cough",
            "symptoms": ["cough"],
            "dosha_involvement": ["kapha"],  # only kapha
        },
        {
            "row_id": "r2",
            "condition": "dry cough",
            "symptoms": ["cough"],
            "dosha_involvement": ["pitta"],  # matches pitta
        },
    ]
    # Request only vata, pitta
    matches = find_matches(mock_rows, doshas=["pitta"], symptom="dry cough", top_k=3)
    assert len(matches) == 1
    assert matches[0][0]["row_id"] == "r2"


def test_find_hybrid_matches():
    from unittest.mock import MagicMock
    import numpy as np
    from src.agent1.query import find_hybrid_matches
    from src.common.llm_client import LLMClient

    mock_client = MagicMock(spec=LLMClient)
    mock_client.embed.return_value = [[1.0, 0.0]]

    rows = [
        {"row_id": "r1", "condition": "cough", "dosha_involvement": ["vata"]},
        {"row_id": "r2", "condition": "fever", "dosha_involvement": ["pitta"]},
    ]
    # Matrix with r1 matching query vector [1.0, 0.0]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    id_map = ["r1", "r2"]

    parsed = {"doshas": ["vata"], "symptom": "cough"}
    results = find_hybrid_matches(
        parsed, rows, matrix=matrix, id_map=id_map, client=mock_client, top_k=2
    )

    assert len(results) == 1
    top_row, blended_score, breakdown = results[0]
    assert top_row["row_id"] == "r1"
    assert blended_score >= 0.8
    assert "semantic" in breakdown
    assert "fuzzy" in breakdown

