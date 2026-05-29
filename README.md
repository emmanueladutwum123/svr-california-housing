# Optimised Support Vector Regression for California Housing Price Prediction

**The Critical Role of Feature Engineering and Hyperparameter Tuning**

> Under Revision · Targeting Discover Artificial Intelligence (Springer Nature) · 2026  
> Author: Emmanuel Adutwum — Soka University of America

---

## Overview

Preethi et al. (2025) reported SVR ranking last among regression models on the California Housing benchmark with R² = 0.60. This paper shows that result reflects missing preprocessing — not an inherent algorithmic limitation.

With a proper leakage-safe pipeline, domain-motivated feature engineering, and systematic hyperparameter tuning, SVR-RBF achieves **R² = 0.723** — a **+0.123 absolute gain (~20% relative improvement)**.

---

## Key Results

| Stage | R² | Gain |
|---|---|---|
| Baseline (no scaling) | −0.054 | — |
| + Feature scaling | 0.690 | +0.744 |
| + Feature engineering | 0.716 | +0.026 |
| + Hyperparameter tuning | **0.723** | +0.008 |

**10-fold cross-validation:** mean R² = 0.703, 95% CI [0.630, 0.775]

**10-model comparison:** SVR ranks 4th — below XGBoost (0.832), Random Forest (0.814), Gradient Boosting (0.783); substantially above simpler baselines.

---

## Methodology

- **Feature engineering:** 10 domain-motivated features derived from 8 raw inputs (rooms per household, bedrooms ratio, population density, geographic clusters, log-income, etc.)
- **Feature selection:** Ensemble of Mutual Information (40%), Pearson correlation (30%), Random Forest importance (30%) — top 12 features selected
- **Pipeline:** Leakage-safe scikit-learn `Pipeline` with `ColumnTransformer` + `SVR` (scaling recomputed per CV fold)
- **Tuning:** `RandomizedSearchCV` — 20 iterations, 3-fold inner CV
- **Ablation study:** Formal 4-stage isolation of each pipeline component's contribution

---

## Repository Contents

| File | Description |
|---|---|
| `openwork.tex` | Main LaTeX source (Springer Nature format) |
| `references.bib` | Bibliography |
| `sn-jnl.cls`, `sn-mathphys-num.bst`, `openwork.sty` | Springer Nature class files |
| `Picture1.png` – `Picture13.png` | All paper figures |
| `svr_implementation.py` | Full corrected Python implementation |
| `svr_notebook.ipynb` | Jupyter notebook with all outputs |
| `Optimised_SVR_Paper_REVISED.pdf` | Compiled PDF (version submitted to Springer/arXiv) |

---

## Submission History

| Date | Journal | Status |
|---|---|---|
| May 2026 | Springer Nature — Machine Learning | Rejected (scope: applied domain paper; missing CIS) |
| May 2026 | Discover Artificial Intelligence (Springer Nature) | Under revision / resubmission |

## Corrections in This Version (May 18, 2026)

Applied before Springer/arXiv submission:
- Fixed `\keywords{}` macro formatting (was `\textbf{Index Terms:}`)
- Added figure cross-references: Fig. 11 (CV folds), Fig. 12 (model comparison), Fig. 13 (ablation)

---

## Dataset

California Housing dataset — 20,640 census block groups, 8 features, median house value target.  
Source: Pace & Barry (1997) via `sklearn.datasets.fetch_california_housing`.

---

## Citation

```
Adutwum, E. (2026). Optimised Support Vector Regression for California Housing Price
Prediction: The Critical Role of Feature Engineering and Hyperparameter Tuning.
Submitted to SN Computer Science, Springer Nature.
```
