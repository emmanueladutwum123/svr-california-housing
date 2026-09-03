# A Controlled Reassessment of Support Vector Regression on the California Housing Benchmark

**with Feature Scaling, Feature Selection and Hyperparameter Tuning**

> Major revision · *Discover Artificial Intelligence* (Springer Nature) · submission `6cf4e30e-1a5b-4ad2-99e5-2320f0737507`
> Author: Emmanuel Adutwum — Soka University of America

---

## What this repository contains

Everything needed to reproduce the paper: one script per experiment, the JSON record each
one writes, and the generators that turn those records into every number, table and figure
in the manuscript. **No numerical value is typed by hand anywhere in the paper.**

```
common.py              shared loading, scaling, selection, pipeline construction
e1_replication.py      v1 replication
e1b_preethi_reproduction.py   direct reproduction of the prior configuration
e2_fair_comparison.py  equal-budget comparison of all 10 models
e3_statistics.py       corrected intervals + Nadeau-Bengio paired tests
e4_ablation.py         single-path ablation (the v1 protocol, for contrast)
e4b_extended_ablation.py      full 4-factor factorial + Shapley attribution
e5_validation.py       spatial blocking, selection inside folds, nested CV
e6_datasets.py         King County and Ames replication
e7_sensitivity.py      sweeps of every hand-chosen constant
e8_kernels_search_size.py     kernels, grid vs random search, training-set size
make_numbers.py        results/*.json -> paper/numbers.tex
make_tables.py         results/*.json -> paper/tables/*.tex
make_figures.py        results/*.json -> paper/figures/*.pdf
build_paper.sh         regenerate everything, then compile all four documents
```

Run `./build_paper.sh` to rebuild the manuscript, response letter, contribution information
sheet and cover letter from the result records.

---

## Corrections to the previous version

This revision withdraws three claims made in the earlier version of this work. They are
listed here because the repository is cited in the paper's data availability statement, and
anyone arriving from the preprint should see them.

**1. The reference value was misread.** The $R^2 \approx 0.60$ previously attributed to the
SVR of Preethi et al. (2025) is the score of their *linear, ridge and polynomial-ridge*
models. Their SVR-RBF result is approximately **0.15**. Reproducing the configuration they
describe, over 5 seeds:

| Configuration | Test R² | Test MSE |
|---|---|---|
| Unscaled, library defaults | −0.020 ± 0.010 | 1.343 |
| Unscaled, tuned | 0.514 ± 0.010 | 0.640 |
| Standardised, defaults | 0.737 ± 0.008 | 0.346 |
| Standardised, tuned | 0.766 ± 0.011 | 0.309 |

The two unscaled configurations bracket their reported result on both metrics.

**2. The "95.7% of the gain comes from scaling" figure was an artefact of ablation order.**
Evaluating all 32 cells of the four-factor design and attributing by Shapley value over all
24 orderings:

| Component | Apparent contribution (range over orderings) | Shapley value |
|---|---|---|
| Scaling | −0.024 to +0.720 | **0.272** |
| Tuning | range 0.516 | 0.188 |
| Selection | — | 0.179 |
| Derived features | — | 0.151 |

Scaling remains the largest single contributor, but at roughly a third of the total rather
than 95.7%.

**3. The confidence interval formula was wrong.** The earlier version reported
`mean ± 1.96·SD`, a prediction interval for one fold rather than a confidence interval for
the mean. Over 10 identical splits the tuned SVR gives 0.7757 ± 0.0099, with the correct
interval [0.769, 0.783], the Nadeau–Bengio interval [0.759, 0.792], and the earlier
expression [0.756, 0.795]. Models are now compared with the Nadeau–Bengio corrected
resampled *t*-test on identical splits; in 2 of 21 pairwise comparisons the correction
changes a significant result to a non-significant one.

---

## Two negative results

Reported rather than omitted, because both contradict the earlier framing:

- **Feature-specific scaling does not beat plain standardisation.** On an identical split the
  hand-assigned strategy gives 0.6925 against 0.7453 for uniform standardisation, and the
  difference is significant under a paired test.
- **The derived features contribute little** once selection and tuning are in place.

---

## Scope

All conclusions are restricted to the datasets, feature sets, search spaces and
implementations described in the manuscript. That feature scaling matters for distance-based
kernels is long established and is not claimed here as a finding.

## Data

- California Housing — `sklearn.datasets.fetch_california_housing`
- King County house sales — [OpenML 42731](https://www.openml.org/d/42731)
- Ames Housing — [OpenML 42165](https://www.openml.org/d/42165)

## Previous version

The v1 sources (`openwork.tex`, `Picture*.png`, `svr_implementation.py`,
`svr_notebook.ipynb`, `CIS_Discover_AI.tex`) remain at the repository root and in the git
history. They correspond to the preprint arXiv:2605.08660 and carry the claims corrected
above.
