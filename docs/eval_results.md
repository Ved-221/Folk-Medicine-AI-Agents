# Retrieval & Extraction Evaluation Results

This document tracks empirical evaluation benchmarks on the ground-truth Gold Set (`data/gold/`).

## 1. Retrieval Accuracy Benchmarks (Gold Queries = 30)

| Method | Top-1 Accuracy | Top-3 Accuracy | No-Match Correctness | Evaluated Queries |
|---|---|---|---|---|
| **Fuzzy only (Baseline Task 1.4)** | **25.0%** (6/24) | **37.5%** (9/24) | **16.7%** (6/6) | 30 |
| **Hybrid Semantic + Fuzzy (Phase 4)** | **29.2%** (7/24) | **41.7%** (10/24) | **16.7%** | 30 |

### Accuracy Breakdown by Query Category (Fuzzy Baseline)

| Category | Queries | Top-1 Correct | Top-3 Correct | Top-1 % |
|---|---|---|---|---|
| `colloquial` | 5 | 0 | 2 | 0.0% |
| `elderly` | 1 | 0 | 0 | 0.0% |
| `hinglish` | 5 | 1 | 1 | 20.0% |
| `missing_field` | 2 | 0 | 0 | 0.0% |
| `pregnancy_caution` | 1 | 1 | 1 | 100.0% |
| `red_flag` | 5 | 0 | 0 | 0.0% |
| `standard` | 6 | 4 | 4 | 66.7% |
| `typo` | 5 | 0 | 1 | 0.0% |

---

## 2. Extractor Benchmarks (Phase 3)

| Metric | Target | Baseline (v1) | Achieved (v2) |
|---|---|---|---|
| Hallucination Rate (unquoted claims) | 0.0% | N/A | **0.0%** |
| Symptom Term Recall | ≥ 80.0% | N/A | **100.0%** (Prec: 100.0%) |
| Ingredient Recall | ≥ 80.0% | N/A | **100.0%** (Prec: 100.0%) |
| Pregnancy Caution Recall | ~100.0% | N/A | **100.0%** |
| Child Caution Recall | ~100.0% | N/A | **100.0%** |
| Share Routed to Review Queue | - | N/A | **0.0%** |
