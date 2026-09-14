import pytest
from eval_harness import Direction
from eval_harness.metrics import get_metric

from review_peft.eval_metrics import (
    EMPTY_RATE,
    LENGTH_RATIO,
    MICRO_F1,
    register_task_metrics,
)


def test_registration_is_idempotent():
    assert register_task_metrics() == register_task_metrics()


def test_quality_metrics_are_higher_is_better():
    assert get_metric(MICRO_F1).direction is Direction.HIGHER_IS_BETTER


def test_failure_metrics_are_lower_is_better():
    assert get_metric(LENGTH_RATIO).direction is Direction.LOWER_IS_BETTER
    assert get_metric(EMPTY_RATE).direction is Direction.LOWER_IS_BETTER


def test_length_ratio_flags_a_model_that_copies_its_input():
    """Similarity metrics reward copying; this is what catches it."""
    copier = [{"summary_words": 100, "source_words": 100}]
    real = [{"summary_words": 12, "source_words": 100}]
    assert get_metric(LENGTH_RATIO)(copier) == pytest.approx(1.0)
    assert get_metric(LENGTH_RATIO)(real) == pytest.approx(0.12)


def test_empty_rate_catches_a_model_degrading_into_silence():
    """Empty predictions are excluded from BERTScore, so this must be tracked."""
    observations = [{"empty": True}, {"empty": False}]
    assert get_metric(EMPTY_RATE)(observations) == pytest.approx(0.5)


def test_micro_f1_counts_unreadable_output_as_wrong():
    observations = [
        {"prediction": None, "reference": "a"},
        {"prediction": "a", "reference": "a"},
    ]
    assert get_metric(MICRO_F1)(observations) == pytest.approx(0.5)


def test_metrics_of_nothing_are_none_not_zero():
    assert get_metric(LENGTH_RATIO)([]) is None
    assert get_metric(EMPTY_RATE)([]) is None
    assert get_metric(MICRO_F1)([]) is None
