# Model card — ChurnLens demo model

Static summary of the committed demo model. The app's *About / Model Card* page shows the same information
live from `models/metadata.json`, which is authoritative; if the two ever disagree, trust the metadata.

> **Analysis aid, not a validated production system.** Trained on a public sample data set. Nothing here
> demonstrates value for any real company, and no financial gain has been measured.

## Model details

| | |
|---|---|
| Task | Binary classification: probability that a telecom customer churns |
| Selected model | Logistic regression (L2, `C = 1`), inside one scikit-learn `Pipeline` (Telco feature engineering → imputation / scaling / one-hot → classifier) |
| Selection | Highest mean CV average precision, simplest model within one standard error (`one_se` rule) among 4 learners + a dummy baseline |
| Training data | 5 634 of the 7 043 rows of the IBM Telco sample data (stratified 80/20 split, seed 42) |
| Input | 19 raw columns (see the input contract in the app or `metadata.json → schema`) |
| Output | Churn probability, predicted class at a threshold, risk segment (Low < 0.20 ≤ Medium < 0.50 ≤ High) |
| Decision threshold | 0.33 — maximises F1 on out-of-fold training predictions (test set not used) |
| Environment | Python 3.14.5, scikit-learn 1.8.0, pandas 3.0.3, NumPy 2.4.6 (recorded in the metadata; the loader refuses another scikit-learn version) |
| Reproducibility | Fixed seeds; `python scripts/verify_reproducibility.py` regenerates and compares every report |

## Intended use

Exploring a churn-analysis workflow: comparing models, studying errors and thresholds, testing retention scenarios
under explicit assumptions, and demonstrating engineering practice. **Out of scope:** automated decisions about
individuals, credit/eligibility/pricing decisions, any use on data that does not follow the input contract, any claim
of business impact.

## Data

See [DATA_CARD.md](DATA_CARD.md). In short: 7 043 customers, 26.5 % churned, snapshot without dates, licence not
verified (file not redistributed).

## Performance (what was actually measured)

Validation — 5-fold stratified cross-validation on the training split (mean ± std over folds):

| Model | ROC-AUC | Average precision | Log loss | Brier |
|---|---|---|---|---|
| Dummy (class prior) | 0.500 ± 0.000 | 0.265 ± 0.000 | 0.579 | 0.195 |
| **Logistic regression (selected)** | **0.848 ± 0.010** | **0.667 ± 0.020** | **0.414** | **0.134** |
| Decision tree | 0.827 ± 0.008 | 0.611 ± 0.023 | 0.505 | 0.142 |
| Random forest | 0.845 ± 0.011 | 0.661 ± 0.026 | 0.416 | 0.135 |
| Hist. gradient boosting | 0.844 ± 0.009 | 0.664 ± 0.020 | 0.417 | 0.136 |

The top four are within fold-to-fold noise of each other. (Full table, all nine metrics, timings and per-fold
values: `reports/model_comparison.csv`, `reports/cv_folds.csv`.)

Held-out test set (1 409 customers, **used once**, selected model only) with 95 % bootstrap intervals over customers:

| Metric @ threshold 0.33 | Value | 95 % interval |
|---|---|---|
| ROC-AUC | 0.842 | 0.823 – 0.865 |
| Average precision | 0.634 | 0.584 – 0.689 |
| Precision | 0.539 | 0.496 – 0.582 |
| Recall | 0.725 | 0.681 – 0.773 |
| F1 | 0.618 | 0.583 – 0.655 |

At threshold 0.5: precision 0.655, recall 0.532, F1 0.587. At 0.33 the model flags 503 of the 1 409 test customers
(36 %) and catches 72 % of those who churned; about 46 % of the flagged customers did not churn (false alarms).

Calibration (out-of-fold): mean predicted churn probability 0.2654, observed rate 0.2654, expected calibration error
0.011. Test-set risk segments: Low 6.6 %, Medium 33.5 %, High 65.5 % observed churn.

Most influential inputs by test-set permutation importance (drop in average precision): `tenure` (0.20),
`InternetService` (0.07), `MonthlyCharges` (0.05), `Contract` (0.04); everything else ≤ 0.01. These are associations
the model uses, not causes.

## Limitations and risks

* **Sample data, one provider.** Results do not transfer to other companies, markets or periods without
  revalidation.
* **Snapshot without dates.** `tenure` and `TotalCharges` are cumulative values that partly encode the outcome for
  customers who left; no temporal validation was possible. Real use would need features available *before* the
  prediction date.
* **Probabilities are calibrated on this data only.** Retention economics that sum probabilities inherit any
  miscalibration on a new population.
* **Single split, fixed hyperparameters.** Differences smaller than the CV standard deviation are not meaningful.
  The test intervals above capture sample noise only.
* **Sensitive attributes** (`gender`, `SeniorCitizen`) are inputs; no fairness or subgroup-performance audit was
  performed. Do not use the scores on individuals without one.
* **Model file = pickle.** Load only files you trained; `metadata.json` carries a SHA-256 for integrity (not
  authenticity) and the scikit-learn version it needs.
* A model trained from the app on a user's CSV is a *different* model: labelled "Custom", kept in the session
  only, never equivalent to this one.

## Ethical considerations

Churn scores can drive who receives retention offers (or who is ignored). Offers differentiated by sensitive
attributes, or contact decisions that disadvantage groups, would need legal and ethical review that this project
has not done.
