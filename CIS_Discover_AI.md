# Contribution Information Sheet (CIS)
## Discover Artificial Intelligence — Springer Nature

---

**Manuscript Title:**
Optimised Support Vector Regression for California Housing Price Prediction: The Critical Role of Feature Engineering and Hyperparameter Tuning

**Corresponding Author:**
Emmanuel Adutwum
Soka University of America, Aliso Viejo, California, United States
eadutwum@soka.edu

**Previous Submission Reference (transfer):**
Submission ID: 80b45e30-56da-4033-b5e6-07f0e94c6143

---

## 1. What is the scientific question addressed?

A recent comparative study (Preethi et al., SN Computer Science, 2025) reported that Support Vector Regression (SVR) ranks last among five regression algorithms on the California Housing benchmark, achieving R² = 0.60 and concluding that SVR is a weak choice for this class of problem.

This paper asks a different question: **Does the reported poor performance reflect an inherent limitation of SVR as an algorithm, or does it reflect the experimental configuration under which SVR was evaluated?**

This is a question about **evaluation methodology and reproducibility in applied AI** — specifically, whether conclusions about algorithm suitability drawn from under-configured experiments are valid, and how much performance is left on the table by omitting standard preprocessing steps.

---

## 2. What methodology is used?

A structured, four-stage experimental workflow is applied entirely within a leakage-safe scikit-learn Pipeline:

- **Stage 1 — Baseline:** SVR-RBF trained on raw, unscaled features (replicating the conditions of Preethi et al., 2025).
- **Stage 2 — Scaling:** StandardScaler applied within the Pipeline, ensuring scaling statistics are recomputed on each cross-validation fold (no data leakage).
- **Stage 3 — Feature Engineering:** Ten domain-motivated features derived from the eight raw inputs (rooms-per-household, bedrooms ratio, population density, log-median income, geographic interaction terms). An ensemble feature importance score combining Mutual Information (40%), Pearson correlation (30%), and Random Forest importance (30%) selects the top 12 most predictive features.
- **Stage 4 — Hyperparameter Tuning:** RandomizedSearchCV with 20 iterations and 3-fold inner cross-validation identifies the optimal kernel, C, epsilon, and gamma configuration.

A formal ablation study isolates the R² contribution of each pipeline component. Ten-fold outer cross-validation provides confidence intervals on the final generalisation estimate.

---

## 3. What are the key findings?

| Pipeline Stage | Test R² | Gain |
|---|---|---|
| Baseline — no scaling (replication of prior work) | −0.054 | — |
| + Feature scaling only | 0.690 | **+0.744** |
| + Domain-motivated feature engineering | 0.716 | +0.026 |
| + Hyperparameter tuning (RandomizedSearchCV) | **0.723** | +0.008 |

**Primary finding:** Feature scaling alone accounts for 95.7% of the total R² improvement (from −0.054 to 0.690). This is not a minor artefact — it is a +0.744 swing that reverses SVR from negative predictive power to competitive performance. Feature engineering and tuning provide further but smaller incremental gains.

**Secondary finding:** The properly configured SVR-RBF achieves R² = 0.723 and ranks 4th in a 10-model comparison (below XGBoost 0.832, Random Forest 0.814, Gradient Boosting 0.783), outperforming Decision Trees, Linear Regression, Ridge, Lasso, Elastic Net, and K-Nearest Neighbours.

**Generalisation:** Ten-fold cross-validation yields mean R² = 0.703 (95% CI [0.630, 0.775]). The confidence interval does not overlap with the baseline R² = 0.60, confirming the improvement is statistically robust across train-test splits.

---

## 4. What are the specific contributions to the AI field?

1. **Methodological contribution — Ablation study design for algorithm evaluation:** The paper provides a replicable four-stage ablation framework for isolating the contribution of individual pipeline components (scaling, feature engineering, hyperparameter optimisation). This framework is applicable beyond SVR to any kernel-based or distance-sensitive algorithm.

2. **Empirical contribution — Quantifying the preprocessing sensitivity of kernel methods:** The +0.744 R² gain from scaling alone on a widely-used benchmark provides a concrete, citable data point on how severely unscaled inputs degrade SVR performance. The result corroborates findings by Cao and Tay (2001) and Fernandez-Delgado et al. (2014) on a modern standardised benchmark.

3. **Reproducibility contribution — Leakage-safe pipeline implementation:** The scikit-learn Pipeline with ColumnTransformer ensures that all preprocessing statistics are computed exclusively on training data within each cross-validation fold. The full implementation is released as open-source code alongside the paper, enabling independent verification.

4. **Evaluation practice contribution — Caution against conclusions from under-configured experiments:** The paper demonstrates that algorithm rankings derived from experiments that omit standard preprocessing can be misleading. This has direct implications for the AI/ML community's evaluation standards, particularly in comparative studies and benchmark publications.

---

## 5. Why is this work significant for Discover Artificial Intelligence?

The scope of Discover Artificial Intelligence explicitly covers *"machine and deep learning"*, *"data analytics"*, and *"AI enabled data-driven techniques"* — all of which are directly addressed by this work.

Beyond the housing application, the paper raises a question that is fundamental to applied AI research: **how much of the perceived performance of an algorithm is a property of the algorithm itself, and how much is a property of the experimental pipeline around it?** The answer here — that 95.7% of SVR's improvement comes from a single preprocessing step — is both striking and broadly generalisable to any practitioner choosing between kernel methods and ensemble methods.

The California Housing dataset is one of the most widely cited benchmarks in supervised learning (20,640+ instances, part of the scikit-learn standard library), making this result immediately interpretable and reproducible by any reader with a Python environment.

---

## 6. Who is the intended audience?

- Machine learning practitioners evaluating SVR and kernel methods for regression tasks
- Researchers designing comparative benchmarking studies who need guidance on fair experimental configuration
- Data science educators using the California Housing dataset in coursework or tutorials
- AI researchers working on automated machine learning (AutoML) and pipeline design, where preprocessing sensitivity is a core concern

---

## 7. Is the research reproducible?

Yes. The full implementation is publicly available at:
**https://github.com/emmanueladutwum123/svr-california-housing**

The repository includes:
- Complete LaTeX source (`openwork.tex`)
- Corrected Python implementation (`svr_implementation.py`)
- Jupyter notebook with all outputs (`svr_notebook.ipynb`)
- Dataset: `sklearn.datasets.fetch_california_housing` (publicly available, no special access required)
- All 13 figures in PNG format

All results reported in the manuscript can be reproduced by running `svr_implementation.py` in any standard Python environment with scikit-learn ≥ 1.0.

---

## 8. Keywords

Support Vector Regression, Feature Engineering, Hyperparameter Optimisation, Ablation Study, Preprocessing Sensitivity, California Housing Dataset, Scikit-learn Pipeline, Kernel Methods, Reproducible Machine Learning, Supervised Regression

---

## 9. Conflict of Interest Statement

The author declares no conflicts of interest.

---

## 10. Data Availability Statement

The California Housing dataset used in this study is publicly available through the scikit-learn Python library (`sklearn.datasets.fetch_california_housing`), originally published by Pace and Barry (1997). No proprietary or restricted data are used. All code and experimental outputs are available at https://github.com/emmanueladutwum123/svr-california-housing.

---

*Prepared for submission to Discover Artificial Intelligence (Springer Nature), 2026.*
