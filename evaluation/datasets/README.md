# Evaluation data

Datasets are **not** committed. The review corpus used to develop this was
provided for a course and is not mine to redistribute.

`review-peft` reads a CSV pointed at by `REVIEW_DATA_PATH`:

| Column | Required | Purpose |
|---|---|---|
| `review` | yes | Free-text customer review |
| `summary` | yes | Reference summary |
| `category` | for the classification baseline | Gold label |

## The split is derived, not stored

There is no `train.csv` / `test.csv` here on purpose. The held-out slice is
computed from `REVIEW_SEED` and `REVIEW_EVAL_SIZE` every time, which is what
guarantees the base and tuned runs score identical rows without depending on two
files staying in sync.

**Changing either value invalidates every baseline committed before it.** The
report records `held-out(seed=…,n=…)` as its dataset string precisely so that
mismatch is visible rather than silent.

## A public substitute

The **Amazon Reviews** and **Yelp Open Dataset** corpora both carry free-text
reviews with ratings; either supports the classification baseline directly.
Reference summaries are the harder half — a small hand-written set of 100 is
enough to make the comparison run, and honest about its size.
