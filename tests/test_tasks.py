from review_peft.generation import ScriptedGenerator
from review_peft.tasks import Classifier, Summarizer


def test_summarizer_records_the_variant():
    """Every observation carries which model produced it."""
    result = Summarizer(ScriptedGenerator(["a summary"]), variant="tuned").predict(
        "x y"
    )
    assert result["variant"] == "tuned"


def test_summarizer_measures_compression():
    result = Summarizer(ScriptedGenerator(["two words"])).predict("one two three four")
    assert result["summary_words"] == 2
    assert result["source_words"] == 4


def test_summarizer_flags_empty_output():
    assert Summarizer(ScriptedGenerator([""])).predict("x")["empty"] is True
    assert Summarizer(ScriptedGenerator(["ok"])).predict("x")["empty"] is False


def test_summarizer_carries_the_reference_when_given_one():
    result = Summarizer(ScriptedGenerator(["s"])).predict("x", reference="gold")
    assert result["reference"] == "gold"


def test_base_and_tuned_are_prompted_identically():
    """The comparison must measure the model, not the scaffolding."""
    base, tuned = ScriptedGenerator(["a"]), ScriptedGenerator(["b"])
    Summarizer(base, variant="base").predict("the same review")
    Summarizer(tuned, variant="tuned").predict("the same review")
    assert base.prompts == tuned.prompts


def test_classifier_extracts_a_label():
    result = Classifier(ScriptedGenerator(["Category: positive"])).predict("x")
    assert result["prediction"] == "positive"


def test_classifier_returns_none_on_unreadable_output():
    assert Classifier(ScriptedGenerator(["no idea"])).predict("x")["prediction"] is None
