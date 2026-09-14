import pytest

from review_peft.data import Record, train_eval_split
from review_peft.exceptions import DatasetError


def test_split_sizes(settings, records):
    train, held_out = train_eval_split(records, settings)
    assert len(held_out) == settings.eval_size
    assert len(train) == len(records) - settings.eval_size


def test_no_overlap_between_train_and_held_out(settings, records):
    """The eval rows must never be trained on, or the comparison is worthless."""
    train, held_out = train_eval_split(records, settings)
    assert {r.review for r in train}.isdisjoint({r.review for r in held_out})


def test_split_is_deterministic(settings, records):
    """The base and tuned runs happen days apart; the slice must not move."""
    first = [r.review for r in train_eval_split(records, settings)[1]]
    second = [r.review for r in train_eval_split(records, settings)[1]]
    assert first == second


def test_a_different_seed_produces_a_different_slice(records):
    from review_peft.settings import Settings

    a = train_eval_split(records, Settings(_env_file=None, seed=1, eval_size=10))[1]
    b = train_eval_split(records, Settings(_env_file=None, seed=2, eval_size=10))[1]
    assert [r.review for r in a] != [r.review for r in b]


def test_split_actually_shuffles(settings, records):
    """An unshuffled split would hand back whatever order the file had."""
    held_out = train_eval_split(records, settings)[1]
    assert [r.review for r in held_out] != [r.review for r in records[:10]]


def test_empty_dataset_is_rejected(settings):
    with pytest.raises(DatasetError, match="empty"):
        train_eval_split([], settings)


def test_dataset_smaller_than_the_eval_slice_is_rejected(settings):
    with pytest.raises(DatasetError, match="nothing would be left to train on"):
        train_eval_split([Record(review="r", summary="s")] * 5, settings)


def test_missing_file_names_the_path(tmp_path):
    from review_peft.data import load_records

    with pytest.raises(DatasetError, match="not found"):
        load_records(tmp_path / "absent.csv")


def test_missing_required_column_is_named(tmp_path):
    from review_peft.data import load_records

    path = tmp_path / "reviews.csv"
    path.write_text("review,category\nhello,a\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="summary"):
        load_records(path)


def test_loads_a_well_formed_csv(tmp_path):
    from review_peft.data import load_records

    path = tmp_path / "reviews.csv"
    path.write_text("review,summary,category\nhello there,short,a\n", encoding="utf-8")
    loaded = load_records(path)
    assert loaded[0].review == "hello there"
    assert loaded[0].summary == "short"
