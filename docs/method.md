# Method notes

## The experiment

| | Base run | Tuned run |
|---|---|---|
| Model | 4-bit base, untouched | Same base + trained LoRA adapter |
| Rows scored | Held-out slice, seed 42 | **The same** held-out slice, seed 42 |
| Prompt | `sft_text(review)` | `sft_text(review)` — identical |
| Decoding | Greedy | Greedy |
| Scorer | Same metric set | Same metric set |

Exactly one thing differs. That is what makes the delta attributable.

## What would invalidate it, and what stops each

**Reshuffling between runs.** `train_eval_split` is seeded and pure, and the
report records `held-out(seed=…,n=…)` as its dataset string, so two reports from
different slices are visibly incomparable rather than silently averaged.

**Eval rows in the training set.** `build_training_texts` derives the training
set from the same split function, and `test_training_texts_exclude_the_held_out_slice`
asserts the disjointness with no GPU involved.

**Different prompts.** Training and inference both go through `sft_text`, with
the completion empty at inference. `test_training_and_inference_share_one_template`
pins it.

**Different decoding.** `GENERATION_KWARGS` is module-level and shared. Greedy,
because sampling noise between two runs is indistinguishable from a real effect
at these sample sizes.

**Scoring prompt plus completion.** Decoding returns both. `strip_prompt` takes
the text after the last `### Response:` marker; scoring the pair would inflate
every similarity metric, because the reference overlaps heavily with the prompt
that contained the review.

## Why three summarization metrics, not one

BERTScore alone fails in two directions that matter:

1. **It rewards copying.** Returning the review verbatim scores well. A model
   that learns this during fine-tuning looks like a success.
   → `summary_length_ratio`, which drifts toward 1.0 exactly then.

2. **It ignores what it cannot score.** Empty predictions drop out of the
   average, so a model degrading into silence posts a *rising* BERTScore on a
   shrinking sample.
   → `empty_output_rate`, tracked separately.

Neither guard is a quality score. Both exist because the quality score has a
blind spot pointing at the most likely failure modes of the thing being tested.

## Why the CLI enforces an order

`evaluate --variant base` writes a report. `finetune` trains. `evaluate
--variant tuned --baseline <that report>` compares.

Running the tuned evaluation without a recorded base score produces a number
with nothing to compare it against — which is the shape almost every fine-tuning
write-up takes, and the reason so few of them establish anything.

## What would make this rigorous

1. Commit a public review dataset and switch on the CI gate.
2. Three training seeds, reporting mean and spread on the tuned result. One run
   cannot separate a real gain from initialisation luck.
3. A small hyperparameter sweep over `r` and step count, reported as a table
   rather than a single chosen configuration.
4. Human evaluation on a 50-summary sample, and agreement between the human
   labels and BERTScore. Until that exists, the metric is unvalidated for this
   data.
5. Per-slice breakdown — long versus short reviews, positive versus negative.
   Aggregates hide which kind of review the tuning helped.
6. GPU-hours and token counts alongside latency, so "does fine-tuning help" can
   be answered as a cost question and not only a quality one.
