# Agent 1 Retrieval Layer & Edge-Case Behavior

**Module:** `src/agent1/query.py`  
**Purpose:** Provides deterministic, grounded candidate selection over traditional medicine records.

---

## 1. Retrieval Algorithm

```
User Query: doshas = [d1, d2], symptom = "..."
  │
  ▼
[Deterministic Code Filter]
  Filter rows where: row.dosha_involvement ∩ {d1, d2} ≠ ∅
  │
  ▼
[Rapidfuzz Scoring]
  For each eligible row:
    Compare query with:
      - row.condition (spaces replacing underscores)
      - each item in row.symptoms
      - each item in row.extracted.symptom_terms (if enriched)
    score = max token_set_ratio across all candidate text fields
  │
  ▼
[Threshold Filter]
  Drop any row where score < FUZZY_MATCH_THRESHOLD (default: 60.0)
  │
  ▼
[Ranking & Top-k]
  Sort descending by score and return top-k matches.
```

---

## 2. Edge-Case Specifications

### 2.1 No Match Found
- When no candidate row achieves a score $\ge$ `FUZZY_MATCH_THRESHOLD` (e.g. nonsense input or completely unrelated queries like `"quantum teleportation cybernetic glitch"`):
  - Returns empty list `[]`.
  - Downstream Agent 2 catches this and produces `status: "no_match"` with a structured explanation, avoiding false medical advice.

### 2.2 Tridoshic Input (`["vata", "pitta", "kapha"]`)
- Matches any row that has any dosha involvement.
- Ranking is strictly driven by symptom match score.

### 2.3 Single vs Multi-Dosha Query
- A condition that involves `["pitta"]` will never match a query that only specifies `["vata"]`, enforcing Ayurvedic dosha specificity.
- Dosha filtering runs purely in deterministic Python code prior to similarity scoring.

### 2.4 Fallback Options
- If `allow_dosha_fallback=True` is passed and no symptom match satisfies the threshold, the system can surface top-ranked constitutional remedies for that dosha, accompanied by an explicit warning flag: `"dosage_origin": "derived"` or `"match_type": "dosha_fallback"`.
