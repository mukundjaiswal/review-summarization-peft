[![CI](https://github.com/mukundjaiswal/review-summarization-peft/actions/workflows/ci.yml/badge.svg)](https://github.com/mukundjaiswal/review-summarization-peft/actions/workflows/ci.yml)

# Review Summarization — QLoRA Fine-Tuning

Customer review classification and summarization, answering two questions in
order:

1. **Can prompting alone do the job?** Few-shot classification with an
   instruction-tuned chat model, scored with Micro-F1.
2. **Does fine-tuning beat it, and by how much?** QLoRA on a 4-bit base model,
   with the same base model scored **before and after** on the **same** held-out
   rows.

This is a personal project. It is not derived from any employer's systems, data
or code.

---

## The part that matters

A fine-tuned model that scores well proves nothing on its own.

"Fine-tuning helped" is a **comparative** claim, and it requires the untuned
model measured on identical data with an identical scorer and identical
prompts. That control run is `review-peft evaluate --variant base`, and the
order of operations in the CLI enforces it: you record the base score, then you
train, then you score the tuned model against the committed base report.

Three things in this repo exist only to protect that comparison:

| Guard | Where | What it prevents |
|---|---|---|
| Seeded, deterministic split | `data.train_eval_split` | The two runs scoring different rows and reporting it as a model difference |
| One template for training and inference | `prompts.sft_text` | The tuned model being asked a differently-shaped question than it was taught on |
| Both variants behind one `TextGenerator` | `generation.py` | The gap including a difference in scaffolding |

Each is a test, not a comment. `test_training_texts_exclude_the_held_out_slice`
is the one I would look at first: the most common silent failure in a
fine-tuning project is eval rows leaking into training, and it needs no GPU to
check.

## Why BERTScore is rescaled, and why it is not enough

Raw BERTScore compresses into a narrow band near 0.85, where a real improvement
and rounding error look identical. `rescale_with_baseline=True` spreads the
range so a difference is visible.

But rescaled or not, **similarity metrics reward copying.** A model that returns
the review verbatim posts an excellent BERTScore while having summarized
nothing. So two guards run alongside it:

| Metric | Direction | What it catches |
|---|---|---|
| `bertscore_f1_rescaled` | higher | Summary quality against the reference |
| `summary_length_ratio` | lower | A model that copies its input — drifts toward 1.0, and no similarity metric shows it |
| `empty_output_rate` | lower | A model degrading into silence. Empty predictions are *excluded* from BERTScore, so a failing model can post a **rising** score on a shrinking sample |
| `review_micro_f1` | higher | The prompting-only classification baseline |
| `p50` / `p95` latency | lower | From the harness |

`empty_output_rate` is the subtle one. Without it, the metric that is supposed
to detect failure moves in the wrong direction during exactly the failure it
should catch.

## Why QLoRA rather than full fine-tuning

A 4-bit base plus low-rank adapters fits a 7B model on a single consumer GPU.
Full fine-tuning does not. For a task this narrow, full fine-tuning would cost
more than the task is worth — and the measured question is whether tuning helps
at all, which QLoRA answers at a fraction of the price.

---

## How it fits with my other projects

Scoring, aggregation across seeds, the report format and the regression gate
come from **[eval-harness](https://github.com/mukundjaiswal/eval-harness)**, a
separate package shared with my other two projects.

What belongs to *this* project is one file — `src/review_peft/eval_metrics.py`.
BERTScore is not in the harness on purpose: it pulls in PyTorch, and a harness
that installs PyTorch to give you a regression gate is a harness nobody adopts.

| Project | What it demonstrates |
|---|---|
| [eval-harness](https://github.com/mukundjaiswal/eval-harness) | The shared evaluation layer |
| [financial-complaint-triage](https://github.com/mukundjaiswal/financial-complaint-triage) | Prompting only. Where measured quality actually comes from |
| **this repo** | Fine-tuning, and how to make the comparison honest |
| [agentic-rag-assistant-nutrition](https://github.com/mukundjaiswal/agentic-rag-assistant-nutrition) | Agentic retrieval with LLM-as-judge quality gates |

---

## Install and run

```bash
git clone https://github.com/mukundjaiswal/review-summarization-peft.git
cd review-summarization-peft

python -m venv .venv && source .venv/bin/activate
make install-dev          # installs the shared harness first
pytest                    # no GPU, no weights, no download
```

The training stack is an optional extra, so CI and the test suite never pull in
PyTorch:

```bash
pip install -e ".[train,scoring]"
cp .env.example .env
```

The experiment, in the order that makes it valid:

```bash
review-peft evaluate --variant base  --out evaluation/baselines/base.json   # control
review-peft finetune                                                         # train
review-peft evaluate --variant tuned --baseline evaluation/baselines/base.json
```

The last command exits non-zero if a metric moved the wrong way past tolerance —
including `summary_length_ratio` rising, which is how "the tuned model learned
to copy" shows up as a failed build rather than a celebrated score.

---

## Layout

```
src/review_peft/
  data.py           loading and the seeded split   <- the experimental control
  prompts.py        one template for training and inference
  generation.py     base and tuned behind one interface + a scripted fake
  finetune.py       QLoRA training; dataset construction split out to be testable
  tasks.py          Summarizer and Classifier, returning scorable observations
  eval_metrics.py   what this task is judged on, registered with eval-harness
  settings.py       typed, validated config; seed and eval_size are load-bearing
  cli.py            finetune / evaluate / config
tests/              offline; the leakage check needs no GPU
```

## What this does not do

- **No committed dataset**, so the CI gate is scaffolded but not switched on.
- **Single training run, single seed.** No variance estimate on the tuned
  result, so a small gap is not distinguishable from noise.
- **No hyperparameter search.** `r=16`, `alpha=16`, 60 steps are reasonable
  defaults, not tuned ones.
- **No human evaluation.** BERTScore plus two guards is better than BERTScore
  alone, and still not the same as someone reading the summaries.
- **No cost accounting.** Latency is measured; GPU-hours and tokens are not.

## Development

```bash
make install-dev
make check          # ruff, mypy --strict, pytest
```

## License

MIT. See [LICENSE](LICENSE).
