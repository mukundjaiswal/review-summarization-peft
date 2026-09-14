"""The metrics this task is judged on.

Registered with the shared ``eval-harness`` package. The harness has no concept
of BERTScore or a length ratio, and should not: those belong to the project that
needs them, and shipping BERTScore inside the harness would mean every consumer
installs PyTorch to get a regression gate.

Registration is an explicit call, not an import side effect. The registry is
global and shared with sibling projects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from eval_harness import Direction
from eval_harness.metrics import Observation, register_metric

BERTSCORE_F1: Final = "bertscore_f1_rescaled"
LENGTH_RATIO: Final = "summary_length_ratio"
EMPTY_RATE: Final = "empty_output_rate"
MICRO_F1: Final = "review_micro_f1"

SUMMARIZATION_METRICS: Final = (
    LENGTH_RATIO,
    EMPTY_RATE,
    "p50_latency_seconds",
    "p95_latency_seconds",
)

#: Summarization metrics plus the reference-based scorer, which needs the
#: ``scoring`` extra installed.
SUMMARIZATION_METRICS_SCORED: Final = (BERTSCORE_F1, *SUMMARIZATION_METRICS)

CLASSIFICATION_METRICS: Final = (
    MICRO_F1,
    "p50_latency_seconds",
    "p95_latency_seconds",
)

_STATE: dict[str, bool] = {"registered": False}


def _paired(observations: Sequence[Observation]) -> tuple[list[str], list[str]]:
    pairs = [
        (str(o["prediction"]), str(o["reference"]))
        for o in observations
        if o.get("reference") is not None and o.get("prediction") is not None
    ]
    if not pairs:
        return [], []
    predictions, references = zip(*pairs, strict=True)
    return list(predictions), list(references)


def bertscore_f1(observations: Sequence[Observation]) -> float | None:
    """Mean BERTScore F1, rescaled against the baseline.

    Rescaling is not cosmetic. Raw BERTScore compresses into a narrow band near
    0.85, where a real improvement and rounding error look the same. Rescaling
    against the empirical baseline spreads the range so a difference is visible.

    Requires the ``scoring`` extra; returns ``None`` when it is absent, so a run
    without it reports the metric as unmeasured rather than failing.
    """
    predictions, references = _paired(observations)
    if not predictions:
        return None
    try:
        import evaluate
    except ImportError:
        return None

    result = evaluate.load("bertscore").compute(
        predictions=predictions,
        references=references,
        lang="en",
        rescale_with_baseline=True,
    )
    scores = result["f1"]
    return float(sum(scores) / len(scores)) if scores else None


def summary_length_ratio(observations: Sequence[Observation]) -> float | None:
    """Mean summary length over source length.

    A guard, not a quality score, and the reason it exists is specific:
    similarity metrics reward copying. A model that returns the review verbatim
    posts a strong BERTScore while having summarized nothing. A ratio drifting
    toward 1.0 is exactly that failure, and no similarity metric will show it.
    """
    ratios = [
        o["summary_words"] / o["source_words"]
        for o in observations
        if o.get("source_words")
    ]
    return sum(ratios) / len(ratios) if ratios else None


def empty_output_rate(observations: Sequence[Observation]) -> float | None:
    """Share of cases where the model returned nothing.

    Tracked separately because an empty prediction is excluded from the
    reference-based score, so a model that degrades into silence can post a
    *rising* BERTScore on a shrinking sample.
    """
    if not observations:
        return None
    return sum(bool(o.get("empty")) for o in observations) / len(observations)


def review_micro_f1(observations: Sequence[Observation]) -> float | None:
    """Micro-F1 for the prompting-only classification baseline."""
    from sklearn.metrics import f1_score

    usable = [o for o in observations if o.get("reference") is not None]
    if not usable:
        return None
    predictions = [o.get("prediction") or "__unparsed__" for o in usable]
    references = [str(o["reference"]) for o in usable]
    return float(f1_score(references, predictions, average="micro"))


def register_task_metrics(*, replace: bool = False) -> tuple[str, ...]:
    """Register this task's metrics and return the summarization set.

    Idempotent, so calling it from the CLI and from a notebook in the same
    process is safe.
    """
    if _STATE["registered"] and not replace:
        return SUMMARIZATION_METRICS

    register_metric(
        BERTSCORE_F1,
        bertscore_f1,
        direction=Direction.HIGHER_IS_BETTER,
        description="Mean BERTScore F1, rescaled against the baseline.",
        replace=True,
    )
    register_metric(
        LENGTH_RATIO,
        summary_length_ratio,
        direction=Direction.LOWER_IS_BETTER,
        description=(
            "Mean summary length over source length. Guards against a model "
            "that copies its input and scores well for it."
        ),
        replace=True,
    )
    register_metric(
        EMPTY_RATE,
        empty_output_rate,
        direction=Direction.LOWER_IS_BETTER,
        description="Share of cases where the model returned nothing.",
        replace=True,
    )
    register_metric(
        MICRO_F1,
        review_micro_f1,
        direction=Direction.HIGHER_IS_BETTER,
        description="Micro-F1 for the prompting-only classification baseline.",
        replace=True,
    )

    _STATE["registered"] = True
    return SUMMARIZATION_METRICS
