"""
tests/test_metrics.py – Unit tests for src/metrics.py.

AGENTS.md Hard Rule 19: these tests MUST pass before any model result is reported.

Run with:
    python -m pytest tests/test_metrics.py -v
"""

import math
import sys
import pathlib

# Make src/ importable when running from the project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from src.metrics import (
    r_precision,
    precision_at_k,
    hit_at_k,
    mrr_at_k,
    average_precision_at_k,
    evaluate,
)

# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

# 10 relevant images, gallery of 50
N = 50
R = 10
ALL_IDS = [str(i) for i in range(N)]
RELEVANT = set(ALL_IDS[:R])      # IDs "0" … "9" are relevant


# ---------------------------------------------------------------------------
# r_precision
# ---------------------------------------------------------------------------

class TestRPrecision:
    def test_perfect_ranking(self):
        """Top-R predictions are all relevant → R-precision = 1.0."""
        ranked = list(RELEVANT) + [str(i) for i in range(R, N)]
        assert r_precision(ranked, RELEVANT) == pytest.approx(1.0)

    def test_all_wrong(self):
        """None of the top-R predictions are relevant → 0.0."""
        ranked = [str(i) for i in range(R, N)] + list(RELEVANT)
        assert r_precision(ranked, RELEVANT) == pytest.approx(0.0)

    def test_half_correct(self):
        """Exactly R/2 correct in top-R → 0.5."""
        ranked = list(RELEVANT)[:R // 2] + [str(i) for i in range(R, R + R // 2)]
        # pad to length ≥ R
        ranked += [str(i) for i in range(R + R // 2, N)]
        assert r_precision(ranked, RELEVANT) == pytest.approx(0.5)

    def test_hand_computed(self):
        """Manually verified: 3 of first 5 relevant → 3/5 = 0.6."""
        ranked = ["0", "1", "99", "2", "98", "3", "4"]  # 0,1,2 in first 5
        relevant = {"0", "1", "2", "3", "4"}
        # first R=5: "0","1","99","2","98" → 3 relevant → 3/5
        assert r_precision(ranked, relevant) == pytest.approx(3 / 5)

    def test_empty_relevant(self):
        """Empty relevant set → 0.0 (not an error)."""
        assert r_precision(["a", "b"], []) == pytest.approx(0.0)

    def test_more_predicted_than_relevant(self):
        """Only first R predictions count; extras are ignored."""
        ranked = list(RELEVANT) + ["extra"] * 100
        assert r_precision(ranked, RELEVANT) == pytest.approx(1.0)

    def test_random_baseline_roughly_r_over_n(self):
        """Random permutation should give ≈ R/N on average."""
        import random
        random.seed(0)
        scores = []
        for _ in range(500):
            perm = ALL_IDS[:]
            random.shuffle(perm)
            scores.append(r_precision(perm, RELEVANT))
        mean = sum(scores) / len(scores)
        assert abs(mean - R / N) < 0.05  # within 5 pp of R/N


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_p5(self):
        ranked = list(RELEVANT)[:5] + [str(i) for i in range(R, N)]
        assert precision_at_k(ranked, RELEVANT, k=5) == pytest.approx(1.0)

    def test_zero_p5(self):
        ranked = [str(i) for i in range(R, R + 5)]
        assert precision_at_k(ranked, RELEVANT, k=5) == pytest.approx(0.0)

    def test_hand_computed_p5(self):
        # 2 relevant in top 5 → P@5 = 0.4
        ranked = ["0", "10", "1", "11", "12"]
        assert precision_at_k(ranked, RELEVANT, k=5) == pytest.approx(2 / 5)

    def test_k_zero(self):
        assert precision_at_k(["0", "1"], RELEVANT, k=0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------

class TestHitAtK:
    def test_hit_at_1_found(self):
        assert hit_at_k(["0", "10", "20"], RELEVANT, k=1) == pytest.approx(1.0)

    def test_hit_at_1_not_found(self):
        assert hit_at_k(["10", "0"], RELEVANT, k=1) == pytest.approx(0.0)

    def test_hit_at_5(self):
        ranked = ["10", "11", "12", "13", "0"]
        assert hit_at_k(ranked, RELEVANT, k=5) == pytest.approx(1.0)

    def test_no_hit_at_5(self):
        ranked = ["10", "11", "12", "13", "14"]
        assert hit_at_k(ranked, RELEVANT, k=5) == pytest.approx(0.0)

    def test_returns_float(self):
        result = hit_at_k(["0"], RELEVANT, k=1)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# mrr_at_k
# ---------------------------------------------------------------------------

class TestMRRAtK:
    def test_first_position(self):
        assert mrr_at_k(["0", "10", "20"], RELEVANT, k=20) == pytest.approx(1.0)

    def test_second_position(self):
        assert mrr_at_k(["10", "0", "20"], RELEVANT, k=20) == pytest.approx(0.5)

    def test_no_hit(self):
        ranked = [str(i) for i in range(R, R + 20)]
        assert mrr_at_k(ranked, RELEVANT, k=20) == pytest.approx(0.0)

    def test_hand_computed(self):
        # First relevant at position 4 → MRR = 1/4
        ranked = ["10", "11", "12", "0"] + [str(i) for i in range(13, 30)]
        assert mrr_at_k(ranked, RELEVANT, k=20) == pytest.approx(1 / 4)

    def test_cutoff_respected(self):
        # Relevant ID only at position 5; k=4 → should return 0.0
        ranked = ["10", "11", "12", "13", "0"]
        assert mrr_at_k(ranked, RELEVANT, k=4) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# average_precision_at_k
# ---------------------------------------------------------------------------

class TestAveragePrecisionAtK:
    def test_perfect(self):
        ranked = list(RELEVANT) + [str(i) for i in range(R, N)]
        assert average_precision_at_k(ranked, RELEVANT, k=20) == pytest.approx(1.0)

    def test_all_wrong(self):
        ranked = [str(i) for i in range(R, R + 20)]
        assert average_precision_at_k(ranked, RELEVANT, k=20) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert average_precision_at_k(["a", "b"], [], k=20) == pytest.approx(0.0)

    def test_hand_computed(self):
        # relevant = {A, B};  ranked = [A, x, B, x, x]
        # Hit at rank 1: P=1/1; hit at rank 3: P=2/3
        # AP@5 = (1/1 + 2/3) / min(5, 2) = (5/3) / 2 = 5/6
        ranked = ["A", "x1", "B", "x2", "x3"]
        relevant = {"A", "B"}
        expected = (1.0 / 1 + 2.0 / 3) / min(5, 2)
        assert average_precision_at_k(ranked, relevant, k=5) == pytest.approx(expected)

    def test_normalisation_by_min_k_R(self):
        # k=3, R=10: normalise by min(3,10)=3
        relevant = set(str(i) for i in range(10))
        ranked = ["0", "1", "2"]
        # 3 hits at positions 1,2,3: sum = 1+1+1 = 3; / 3 = 1.0
        assert average_precision_at_k(ranked, relevant, k=3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate (dataset-level)
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _make_perfect(self):
        gt = {f"q{i}": [str(j) for j in range(i * 10, i * 10 + 10)] for i in range(5)}
        preds = {qid: ids + [str(j) for j in range(500, 520)] for qid, ids in gt.items()}
        return preds, gt

    def test_perfect_all_ones(self):
        preds, gt = self._make_perfect()
        metrics = evaluate(preds, gt)
        assert metrics["r_precision"] == pytest.approx(1.0)
        assert metrics["map@20"] == pytest.approx(1.0)
        assert metrics["p@5"] == pytest.approx(1.0)
        assert metrics["hit@1"] == pytest.approx(1.0)
        assert metrics["hit@5"] == pytest.approx(1.0)
        assert metrics["mrr@20"] == pytest.approx(1.0)

    def test_keys_present(self):
        preds, gt = self._make_perfect()
        metrics = evaluate(preds, gt)
        for key in ("r_precision", "map@20", "p@5", "hit@1", "hit@5", "mrr@20"):
            assert key in metrics

    def test_values_in_range(self):
        import random
        random.seed(1)
        gt = {"q0": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]}
        ranked = ALL_IDS[:]
        random.shuffle(ranked)
        preds = {"q0": ranked}
        metrics = evaluate(preds, gt)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_empty_ground_truth_raises(self):
        with pytest.raises(ValueError):
            evaluate({}, {})

    def test_missing_query_prediction_treated_as_empty(self):
        gt = {"q0": ["0", "1"]}
        preds = {}  # no prediction for q0
        metrics = evaluate(preds, gt)
        assert metrics["r_precision"] == pytest.approx(0.0)
