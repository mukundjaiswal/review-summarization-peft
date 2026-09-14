"""Training-set construction, testable without a GPU.

The thing most likely to be silently wrong in a fine-tuning project is whether
the evaluation rows leaked into training. That is checkable with no model at
all, so it is checked here rather than assumed.
"""

from review_peft.data import train_eval_split
from review_peft.finetune import build_training_texts


def test_training_texts_exclude_the_held_out_slice(settings, records):
    _, held_out = train_eval_split(records, settings)
    texts = build_training_texts(records, settings)
    for record in held_out:
        assert not any(record.summary in t for t in texts)


def test_training_texts_cover_every_training_record(settings, records):
    train, _ = train_eval_split(records, settings)
    texts = build_training_texts(records, settings)
    assert len(texts) == len(train)


def test_training_texts_include_the_gold_completion(settings, records):
    """Without the completion the model has nothing to learn from."""
    texts = build_training_texts(records, settings)
    assert all("### Response:" in t for t in texts)
    assert any("summary" in t.split("### Response:")[1] for t in texts)
