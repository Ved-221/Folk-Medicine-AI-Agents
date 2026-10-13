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
