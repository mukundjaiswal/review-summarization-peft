"""The controlled comparison, end to end, driven by scripted generators.

This is the test that proves both seams: the cross-repo one (eval-harness runs
and gates it) and the experimental one (base and tuned scored on identical rows
through identical code).
"""

from eval_harness import EvalCase, EvaluationRunner, RegressionCheck

from review_peft.data import train_eval_split
from review_peft.eval_metrics import SUMMARIZATION_METRICS, register_task_metrics
from review_peft.generation import ScriptedGenerator
from review_peft.tasks import Summarizer


def report_for(generator, variant, settings, records):
    register_task_metrics(replace=True)
    _, held_out = train_eval_split(records, settings)
    cases = [EvalCase(input=r.review, reference=r.summary) for r in held_out]
    summarizer = Summarizer(generator, variant=variant)
    return EvaluationRunner(
        lambda c: summarizer.predict(c.input, c.reference),
        metrics=SUMMARIZATION_METRICS,
        seeds=[0],
    ).run(cases, dataset=f"held-out(seed={settings.seed})")


def test_both_variants_score_the_same_rows(settings, records):
    """The control. Different rows would make the comparison meaningless."""
    base = report_for(ScriptedGenerator(["a"]), "base", settings, records)
    tuned = report_for(ScriptedGenerator(["b"]), "tuned", settings, records)
    assert base.dataset == tuned.dataset
    assert base.n_cases == tuned.n_cases


def test_a_tuned_model_that_copies_its_input_is_caught(settings, records):
    """The failure a similarity metric alone would reward."""
    concise = report_for(
        ScriptedGenerator(["short summary"]), "base", settings, records
    )
    copier = report_for(
        ScriptedGenerator(responder=lambda p: p), "tuned", settings, records
    )
    result = RegressionCheck(tolerance=0.02).compare(
        copier, concise.to_dict(include_observations=False)
    )
    assert not result.passed
    assert "summary_length_ratio" in {c.name for c in result.regressions}


def test_a_model_degrading_into_silence_is_caught(settings, records):
    """Empty output is excluded from similarity scoring, so it needs its own gate."""
    good = report_for(ScriptedGenerator(["a real summary"]), "base", settings, records)
    silent = report_for(ScriptedGenerator([""]), "tuned", settings, records)
    result = RegressionCheck(tolerance=0.02).compare(
        silent, good.to_dict(include_observations=False)
    )
    assert not result.passed
    assert "empty_output_rate" in {c.name for c in result.regressions}


def test_an_equivalent_model_does_not_trip_the_gate(settings, records):
    """A gate that fires on everything is not a gate."""
    a = report_for(ScriptedGenerator(["short summary"]), "base", settings, records)
    b = report_for(ScriptedGenerator(["brief overview"]), "tuned", settings, records)
    result = RegressionCheck(tolerance=0.02).compare(
        b, a.to_dict(include_observations=False)
    )
    assert result.passed


def test_every_observation_records_which_variant_produced_it(settings, records):
    report = report_for(ScriptedGenerator(["s"]), "tuned", settings, records)
    assert {o["variant"] for o in report.observations} == {"tuned"}
