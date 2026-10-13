"""Unit test for evaluation baseline suite."""

from src.eval.eval_retrieval import evaluate_fuzzy_retrieval


def test_evaluate_fuzzy_retrieval():
    results = evaluate_fuzzy_retrieval()
    assert results["total_queries"] == 30
    assert results["retrieval_queries"] == 24
    assert 0.0 <= results["top1_accuracy"] <= 100.0
    assert 0.0 <= results["top3_accuracy"] <= 100.0
    assert "breakdown" in results
