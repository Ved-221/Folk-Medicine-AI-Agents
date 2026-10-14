"""Unit tests for Agent 1 vector embedder and semantic search."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.agent1.embedder import (
    build_index,
    build_row_text_for_embedding,
    load_index,
    search_semantic,
)
from src.common.llm_client import LLMClient


def test_build_row_text_for_embedding():
    row = {
        "condition": "allergic_asthma",
        "symptoms": ["wheezing", "chest tightness"],
        "extracted": {
            "symptom_terms": ["breathlessness"],
            "canonical_ingredients": ["vasaka", "kantakari"],
            "topic_tags": ["respiratory"],
        },
    }
    text = build_row_text_for_embedding(row)
    assert "Condition: allergic asthma" in text
    assert "wheezing" in text
    assert "breathlessness" in text
    assert "vasaka" in text


def test_build_and_load_index(tmp_path):
    mock_client = MagicMock(spec=LLMClient)
    # Return 3 mock 4-dimensional vectors
    mock_client.embed.return_value = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    sample_rows = [
        {"row_id": "r1", "condition": "cough"},
        {"row_id": "r2", "condition": "fever"},
        {"row_id": "r3", "condition": "headache"},
    ]

    matrix, id_map = build_index(sample_rows, index_dir=tmp_path, client=mock_client)
    assert matrix.shape == (3, 4)
    assert id_map == ["r1", "r2", "r3"]

    # Test load_index
    loaded_matrix, loaded_ids = load_index(index_dir=tmp_path)
    assert loaded_ids == ["r1", "r2", "r3"]
    assert np.allclose(matrix, loaded_matrix)


def test_search_semantic_candidate_isolation():
    # Pre-normalized unit vectors
    matrix = np.array([
        [1.0, 0.0],  # r1 (cough)
        [0.8, 0.6],  # r2 (respiratory)
        [0.0, 1.0],  # r3 (fever)
    ], dtype=np.float32)
    id_map = ["r1", "r2", "r3"]

    # Query vector close to r1
    query_vec = [1.0, 0.0]

    # Search with all candidates
    all_results = search_semantic(query_vec, candidate_ids=set(), matrix=matrix, id_map=id_map, top_k=2)
    assert all_results[0][0] == "r1"
    assert pytest.approx(all_results[0][1], 0.01) == 1.0

    # Search isolated to candidate r2 and r3 only (e.g. if r1 had wrong dosha)
    filtered_results = search_semantic(
        query_vec, candidate_ids={"r2", "r3"}, matrix=matrix, id_map=id_map, top_k=2
    )
    assert len(filtered_results) == 2
    assert filtered_results[0][0] == "r2"
    assert "r1" not in [r[0] for r in filtered_results]
