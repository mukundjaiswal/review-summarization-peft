"""The two tasks, each returning an observation the harness can score."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from review_peft import prompts
from review_peft.interfaces import TextGenerator


@dataclass(frozen=True, slots=True)
class Summarizer:
    """Summarizes a review, via whichever variant it was given."""

    generator: TextGenerator
    variant: str = "base"

    def predict(self, review: str, reference: str | None = None) -> dict[str, Any]:
        """Summarize ``review`` and report what the metrics need."""
        summary = self.generator.generate(prompts.sft_text(review)).strip()
        observation: dict[str, Any] = {
            "variant": self.variant,
            "prediction": summary,
            "summary_words": len(summary.split()),
            "source_words": len(review.split()),
            "empty": not summary,
        }
        if reference is not None:
            observation["reference"] = reference
        return observation


@dataclass(frozen=True, slots=True)
class Classifier:
    """Few-shot review classification, the prompting-only baseline."""

    generator: TextGenerator
    examples: Sequence[tuple[str, str]] = ()

    def predict(self, review: str, reference: str | None = None) -> dict[str, Any]:
        """Classify ``review``."""
        raw = self.generator.generate(
            prompts.few_shot_classify_prompt(review, self.examples)
        )
        prediction = _extract_label(raw)
        observation: dict[str, Any] = {
            "raw": raw,
            "prediction": prediction,
            "shots": len(self.examples),
        }
        if reference is not None:
            observation["reference"] = reference
        return observation


def _extract_label(text: str) -> str | None:
    """Pull the label out of a generation, or return ``None``."""
    import re

    match = re.search(r"category\s*[:\-]?\s*([a-z0-9_\- ]+)", str(text).lower())
    if not match:
        return None
    label = match.group(1).strip().split("\n")[0].strip()
    return label or None
