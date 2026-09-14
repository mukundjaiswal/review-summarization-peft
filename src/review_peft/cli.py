"""Command-line interface.

    review-peft evaluate --variant base   --out evaluation/baselines/base.json
    review-peft finetune
    review-peft evaluate --variant tuned  --baseline evaluation/baselines/base.json

That order is the experiment. The control run comes first, on purpose: a tuned
score with no recorded base score is not a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from review_peft import __version__
from review_peft.settings import get_settings

app = typer.Typer(
    name="review-peft",
    help="QLoRA fine-tuning for review summarization, with a controlled comparison.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def config() -> None:
    """Print the resolved configuration, with secrets masked."""
    typer.echo(json.dumps(json.loads(get_settings().model_dump_json()), indent=2))


@app.command()
def finetune(
    max_steps: Annotated[
        int | None, typer.Option(help="Override the configured step count.")
    ] = None,
) -> None:
    """Run the QLoRA fine-tune and write the adapter."""
    from review_peft.finetune import train

    settings = get_settings()
    train(settings, max_steps=max_steps)
    typer.echo(f"Adapter written to {settings.adapter_dir}")


@app.command()
def evaluate(
    variant: Annotated[str, typer.Option(help="base or tuned.")] = "base",
    data: Annotated[
        Path | None, typer.Option(help="CSV of reviews; defaults to REVIEW_DATA_PATH.")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Where to write the report.")] = None,
    baseline: Annotated[
        Path | None, typer.Option(help="Committed baseline to compare against.")
    ] = None,
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds.")] = "0",
    tolerance: Annotated[
        float, typer.Option(help="Allowed drift before a metric counts as regressed.")
    ] = 0.02,
    scored: Annotated[
        bool, typer.Option(help="Include BERTScore; needs the 'scoring' extra.")
    ] = False,
) -> None:
    """Score one variant on the held-out slice.

    The slice is derived from the configured seed, so the base and tuned runs
    score identical rows. That is the control; without it the comparison
    measures a different dataset rather than a different model.
    """
    from eval_harness import EvalCase, EvaluationRunner, RegressionCheck
    from eval_harness.report import load_baseline

    from review_peft.data import load_records, train_eval_split
    from review_peft.eval_metrics import (
        SUMMARIZATION_METRICS,
        SUMMARIZATION_METRICS_SCORED,
        register_task_metrics,
    )
    from review_peft.generation import HuggingFaceGenerator
    from review_peft.tasks import Summarizer

    if variant not in {"base", "tuned"}:
        typer.echo(f"Unknown variant {variant!r}. Use 'base' or 'tuned'.", err=True)
        sys.exit(2)

    register_task_metrics()
    settings = get_settings()
    _, held_out = train_eval_split(load_records(data), settings)

    summarizer = Summarizer(
        HuggingFaceGenerator(variant, settings),  # type: ignore[arg-type]
        variant=variant,
    )
    cases = [EvalCase(input=r.review, reference=r.summary) for r in held_out]

    def task(case: EvalCase) -> dict[str, object]:
        return summarizer.predict(case.input, case.reference)

    report = EvaluationRunner(
        task,
        metrics=SUMMARIZATION_METRICS_SCORED if scored else SUMMARIZATION_METRICS,
        seeds=[int(s) for s in seeds.split(",") if s.strip()],
    ).run(cases, dataset=f"held-out(seed={settings.seed},n={settings.eval_size})")

    destination = out or Path(f"evaluation/results/{variant}.json")
    report.write(destination)
    typer.echo(report.summary())
    typer.echo(f"\nWrote {destination}")

    if baseline is not None:
        result = RegressionCheck(tolerance=tolerance).compare(
            report, load_baseline(baseline)
        )
        typer.echo("")
        typer.echo(result.report())
        if not result.passed:
            typer.echo(
                f"\n{len(result.regressions)} metric(s) moved the wrong way past "
                f"tolerance {tolerance}.",
                err=True,
            )
            sys.exit(1)
        typer.echo("\nNo regression against baseline.")


if __name__ == "__main__":
    app()
