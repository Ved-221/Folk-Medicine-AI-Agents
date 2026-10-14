"""Unit test for evaluation baseline suite."""

from src.eval.eval_retrieval import evaluate_fuzzy_retrieval


def test_evaluate_fuzzy_retrieval():
    results = evaluate_fuzzy_retrieval()
    assert results["total_queries"] == 30
    assert results["retrieval_queries"] == 24
    assert 0.0 <= results["top1_accuracy"] <= 100.0
    assert 0.0 <= results["top3_accuracy"] <= 100.0
    assert "breakdown" in results


def test_evaluate_extractions():
    from src.agent1.loader import load_dataset
    from src.common.config import GOLD_DIR
    from src.eval.eval_extractor import evaluate_extractions

    gold_rows = load_dataset(GOLD_DIR / "gold_rows.jsonl")
    metrics = evaluate_extractions(gold_rows, gold_rows)
    assert metrics["total_evaluated"] == 30
    assert metrics["symptom_recall"] == 100.0
    assert metrics["ingredient_recall"] == 100.0
    assert metrics["hallucination_rate"] == 0.0
