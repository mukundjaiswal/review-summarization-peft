"""Dataset loading and the seeded split that makes the comparison controlled.

The split is the experimental control. Both the base and the tuned run must
score the **same** held-out rows, and the tuned model must never have trained on
them. A reshuffle between runs turns a model comparison into a dataset
comparison without changing a single number's name.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from review_peft.exceptions import DatasetError
from review_peft.prompts import sft_text
from review_peft.settings import Settings, get_settings

REQUIRED_COLUMNS = ("review", "summary")


@dataclass(frozen=True, slots=True)
class Record:
    """One review with its gold summary and optional label."""

    review: str
    summary: str
    category: str | None = None

    def to_sft_text(self) -> str:
        """Render as a single training string."""
        return sft_text(self.review, self.summary)


def load_records(path: str | Path | None = None) -> list[Record]:
    """Load records from CSV.

    Datasets are not committed. Point ``REVIEW_DATA_PATH`` at your own copy.
    """
    import csv

    resolved = Path(path or os.getenv("REVIEW_DATA_PATH") or "data/reviews.csv")
    if not resolved.exists():
        message = (
            f"{resolved} not found. See evaluation/datasets/README.md for the "
            "expected shape."
        )
        raise DatasetError(message)

    with resolved.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        message = f"{resolved} contains no rows"
        raise DatasetError(message)

    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        message = f"{resolved} is missing required columns: {missing}"
        raise DatasetError(message)

    return [
        Record(
            review=r["review"],
            summary=r["summary"],
            category=r.get("category") or None,
        )
        for r in rows
    ]


def train_eval_split(
    records: Sequence[Record], settings: Settings | None = None
) -> tuple[list[Record], list[Record]]:
    """Return ``(train, held_out)``, seeded and disjoint.

    Deterministic by construction: the same seed and the same input always yield
    the same held-out slice, which is what lets a tuned run be compared against a
    base run recorded days earlier.
    """
    import random

    resolved = settings or get_settings()
    if not records:
        message = "Cannot split an empty dataset"
        raise DatasetError(message)
    if len(records) <= resolved.eval_size:
        message = (
            f"Dataset has {len(records)} rows but eval_size is "
            f"{resolved.eval_size}; nothing would be left to train on."
        )
        raise DatasetError(message)

    shuffled = list(records)
    random.Random(resolved.seed).shuffle(shuffled)
    return shuffled[resolved.eval_size :], shuffled[: resolved.eval_size]
