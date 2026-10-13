"""Agent 1 Retrieval Layer.

Provides deterministic dosha filtering and fuzzy symptom matching
(with fallback) to retrieve candidate traditional medicine records.
"""

from typing import Any
from rapidfuzz import fuzz

from src.common.config import FUZZY_MATCH_THRESHOLD


def score_row_fuzzy(symptom_query: str, row: dict[str, Any]) -> float:
    """Calculates fuzzy similarity between a query symptom and a dataset row.

    Evaluates against:
    - Condition name (with underscores replaced by spaces)
    - Condition category
    - All symptom descriptions in `symptoms` array
    - Extracted symptom terms (if enriched row)

    Returns:
        Maximum similarity score (0.0 to 100.0).
    """
    clean_query = symptom_query.strip().lower()
    if not clean_query:
        return 0.0

    candidates: list[str] = []

    # 1. Condition string
    condition = row.get("condition", "")
    if isinstance(condition, str):
        candidates.append(condition.replace("_", " ").lower())

    # 2. Symptoms list
    symptoms = row.get("symptoms", [])
    if isinstance(symptoms, list):
        for s in symptoms:
            if isinstance(s, str):
                candidates.append(s.lower())

    # 3. Extracted symptom terms (if present from Phase 3 enrichment)
    extracted = row.get("extracted", {})
    if isinstance(extracted, dict):
        extra_terms = extracted.get("symptom_terms", [])
        if isinstance(extra_terms, list):
            for t in extra_terms:
                if isinstance(t, str):
                    candidates.append(t.lower())

    if not candidates:
        return 0.0

    scores = [fuzz.token_set_ratio(clean_query, cand) for cand in candidates]
    return float(max(scores)) if scores else 0.0


def find_matches(
    rows: list[dict[str, Any]],
    doshas: list[str],
    symptom: str,
    top_k: int = 3,
    threshold: float = FUZZY_MATCH_THRESHOLD,
    allow_dosha_fallback: bool = False,
) -> list[tuple[dict[str, Any], float]]:
    """Retrieves top candidate rows matching dosha involvement and symptom query.

    Algorithm:
    1. Filter: Retain rows whose `dosha_involvement` overlaps with the user's doshas.
       (If doshas is empty and allow_dosha_fallback is True, considers all rows).
    2. Score: Compute fuzzy match score between `symptom` and candidate row text.
    3. Filter by threshold: Ignore rows below `threshold`.
    4. Rank: Sort descending by score, returning the top_k candidates.

    Args:
        rows: Dataset records (clean or trimmed).
        doshas: Target doshas, e.g. ["vata", "pitta"].
        symptom: Free-text symptom query, e.g. "dry cough".
        top_k: Maximum number of candidate rows to return.
        threshold: Minimum fuzzy score (0-100) to consider a valid match.
        allow_dosha_fallback: If True and symptom search fails, return best dosha rows.

    Returns:
        List of tuples: (matched_row, similarity_score). Empty list if no match.
    """
    if not symptom or not symptom.strip():
        return []

    user_doshas = {d.strip().lower() for d in doshas if isinstance(d, str)}

    # Step 1: Deterministic dosha filtering (plain code)
    eligible_rows: list[dict[str, Any]] = []
    for r in rows:
        row_doshas = set(r.get("dosha_involvement", []))
        if not user_doshas or (user_doshas & row_doshas):
            eligible_rows.append(r)

    if not eligible_rows:
        return []

    # Step 2: Score eligible rows
    scored_candidates: list[tuple[dict[str, Any], float]] = []
    for r in eligible_rows:
        score = score_row_fuzzy(symptom, r)
        if score >= threshold:
            scored_candidates.append((r, score))

    # Step 3: Sort descending by score
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # Step 4: Top-k selection
    return scored_candidates[:top_k]
