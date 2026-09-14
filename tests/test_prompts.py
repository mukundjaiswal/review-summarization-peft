from review_peft.prompts import (
    RESPONSE_MARKER,
    few_shot_classify_prompt,
    sft_text,
    strip_prompt,
)


def test_training_and_inference_share_one_template():
    """If they diverged, the tuned model would be asked a different question."""
    training = sft_text("a review", "the gold summary")
    inference = sft_text("a review")
    assert inference == training.replace("the gold summary", "")


def test_sft_text_has_all_three_sections():
    text = sft_text("a review", "a summary")
    assert "### Instruction:" in text
    assert "### Input:" in text
    assert RESPONSE_MARKER in text


def test_strip_prompt_returns_only_the_completion():
    """Scoring prompt+completion inflates every similarity metric."""
    decoded = sft_text("the review text") + " the generated summary"
    assert strip_prompt(decoded) == "the generated summary"
    assert "the review text" not in strip_prompt(decoded)


def test_strip_prompt_is_safe_on_output_without_the_marker():
    assert strip_prompt("  bare output  ") == "bare output"


def test_strip_prompt_uses_the_last_marker():
    text = f"{RESPONSE_MARKER} first {RESPONSE_MARKER} second"
    assert strip_prompt(text) == "second"


def test_few_shot_examples_precede_the_target():
    prompt = few_shot_classify_prompt("TARGET", [("EXAMPLE", "a")])
    assert prompt.index("EXAMPLE") < prompt.index("TARGET")


def test_zero_shot_classification_prompt_has_no_shots():
    prompt = few_shot_classify_prompt("only this", [])
    assert "only this" in prompt
    assert "Category:" not in prompt.split("only this")[0].split("<</SYS>>")[-1]
