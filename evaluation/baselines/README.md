# Baselines

The base-model report is not documentation here. It is the experimental
control, and it belongs in version control:

```bash
review-peft evaluate --variant base --out evaluation/baselines/base.json
git add evaluation/baselines/base.json
git commit -m "Record base-model baseline on held-out slice (seed 42, n=100)"
```

Then the tuned run is a diff against a committed number:

```bash
review-peft evaluate --variant tuned --baseline evaluation/baselines/base.json
```

Commit the base report **before** training, in its own commit. A baseline
recorded after the fact, from memory or from a rerun, is not a control.

Keep one baseline per configuration and name it for that configuration —
`base.json`, `tuned-r16-s60.json`. Comparing across configurations measures the
configuration change, not a regression.
