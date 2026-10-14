"""Agent 1 Vector Embeddings & Index Layer.

Builds and queries precomputed semantic embedding indices over enriched
traditional medicine records using Gemini embeddings with cosine similarity.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common.config import INDEX_DIR
from src.common.llm_client import LLMClient, default_client


def build_row_text_for_embedding(row: dict[str, Any]) -> str:
    """Creates a concise textual representation of a row optimized for semantic embedding."""
    parts: list[str] = []

    # Condition
    cond = row.get("condition", "").replace("_", " ")
    parts.append(f"Condition: {cond}")

    # Symptoms
    symptoms = row.get("symptoms", [])
    if isinstance(symptoms, list) and symptoms:
        parts.append(f"Symptoms: {', '.join(symptoms)}")

    # Enriched context block (from Phase 3)
    extracted = row.get("extracted", {})
    if isinstance(extracted, dict):
        extra_terms = extracted.get("symptom_terms", [])
        if extra_terms:
            parts.append(f"Symptom terms: {', '.join(extra_terms)}")
        ingredients = extracted.get("canonical_ingredients", [])
        if ingredients:
            parts.append(f"Ingredients: {', '.join(ingredients)}")
        tags = extracted.get("topic_tags", [])
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")

    return ". ".join(parts)


def build_index(
    rows: list[dict[str, Any]],
    index_dir: Path = INDEX_DIR,
    client: LLMClient | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Builds and persists semantic vector index for the provided rows.

    Args:
        rows: Dataset rows (clean or enriched).
        index_dir: Directory to save embeddings.npy and id_map.json.
        client: LLMClient instance.

    Returns:
        Tuple of (normalized_embeddings_matrix, id_map_list).
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    llm = client or default_client
    id_map = [str(r.get("row_id", f"row_{i:03d}")) for i, r in enumerate(rows, start=1)]
    texts = [build_row_text_for_embedding(r) for r in rows]

    raw_vectors = llm.embed(texts)
    matrix = np.array(raw_vectors, dtype=np.float32)

    # Normalize vectors for fast cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_matrix = matrix / norms

    # Save to disk
    np.save(index_dir / "embeddings.npy", normalized_matrix)
    with open(index_dir / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    return normalized_matrix, id_map


def load_index(index_dir: Path = INDEX_DIR) -> tuple[np.ndarray | None, list[str] | None]:
    """Loads precomputed vector index and id map from disk if available."""
    index_dir = Path(index_dir)
    matrix_file = index_dir / "embeddings.npy"
    id_file = index_dir / "id_map.json"

    if not matrix_file.exists() or not id_file.exists():
        return None, None

    try:
        matrix = np.load(matrix_file)
        with open(id_file, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        return matrix, id_map
    except Exception:
        return None, None


def search_semantic(
    query_vector: list[float],
    candidate_ids: set[str],
    matrix: np.ndarray,
    id_map: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Calculates cosine similarity between query vector and eligible candidate rows.

    Args:
        query_vector: Embedding vector of the user query.
        candidate_ids: Subset of row IDs that passed deterministic dosha filtering.
        matrix: Precomputed normalized row embeddings matrix.
        id_map: List of row IDs corresponding to matrix rows.
        top_k: Number of matches to return.

    Returns:
        List of (row_id, cosine_similarity_score) sorted descending by score.
    """
    if matrix is None or not id_map:
        return []

    q_vec = np.array(query_vector, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return []
    q_normed = q_vec / q_norm

    # Compute dot products across all rows
    similarities = np.dot(matrix, q_normed)

    results: list[tuple[str, float]] = []
    for idx, row_id in enumerate(id_map):
        if not candidate_ids or row_id in candidate_ids:
            score = float(similarities[idx])
            results.append((row_id, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
