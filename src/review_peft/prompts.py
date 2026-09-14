"""Prompt and training templates.

One rule governs this module: **the base and tuned variants must be prompted
identically.** If they are not, the comparison measures the difference in
scaffolding rather than the difference in the model, and reports it as if
fine-tuning caused it.

That is why :func:`sft_text` is used to build both the training examples and
the evaluation prompts, with the completion left empty at inference.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

CLASSIFY_SYSTEM_MESSAGE: Final = """\
You are a customer review classification assistant.
Assign the review to exactly one category.

Rules:
- Choose exactly one category from the list provided.
- Respond in the form: Category: <category>
- Do not explain your reasoning.
"""

SUMMARIZE_SYSTEM_MESSAGE: Final = """\
You are a customer review summarization assistant.
Write a short summary of the review.

Rules:
- Use only information present in the review.
- Do not invent product details or sentiment that is not stated.
- Keep the summary to one or two sentences.
"""

CHAT_TEMPLATE: Final = (
    "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_input} [/INST]"
)

#: Single training string per record, as ``SFTTrainer`` consumes it.
SFT_TEMPLATE: Final = """\
### Instruction:
{instruction}

### Input:
{review}

### Response:
{completion}"""

RESPONSE_MARKER: Final = "### Response:"


def chat_prompt(system_message: str, user_input: str) -> str:
    """Instruction-format prompt for the prompting-only baseline."""
    return CHAT_TEMPLATE.format(system_message=system_message, user_input=user_input)


def few_shot_classify_prompt(review: str, examples: Iterable[tuple[str, str]]) -> str:
    """Few-shot classification prompt from ``(review, label)`` pairs."""
    shots = "\n\n".join(
        f"Review: {text}\nCategory: {label}" for text, label in examples
    )
    body = f"{shots}\n\nReview: {review}" if shots else f"Review: {review}"
    return chat_prompt(CLASSIFY_SYSTEM_MESSAGE, body)


def sft_text(review: str, completion: str = "") -> str:
    """Render one record in the fine-tuning format.

    With ``completion`` empty this is the inference prompt, which is exactly the
    point: training and inference share one template, so the tuned model is
    never asked a question shaped differently from the ones it was taught on.
    """
    return SFT_TEMPLATE.format(
        instruction=SUMMARIZE_SYSTEM_MESSAGE.strip(),
        review=review,
        completion=completion,
    )


def strip_prompt(generated: str) -> str:
    """Return only the completion from a full decoded sequence.

    Decoding returns prompt and completion together. Scoring the pair instead
    of the completion inflates every similarity metric, because the reference
    text overlaps heavily with the prompt that contained the review.
    """
    if RESPONSE_MARKER in generated:
        return generated.rsplit(RESPONSE_MARKER, 1)[-1].strip()
    return generated.strip()
