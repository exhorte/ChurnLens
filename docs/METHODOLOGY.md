# Methodology

How models are built, compared and evaluated, and why. Everything here is implemented in
`src/churnlens/` and exercised by `tests/`. Configuration lives in [`configs/config.toml`](../configs/config.toml).

## 1. Protocol

```
labelled data ──► standardise (row-wise only) ──► stratified split 80 / 20 (seed 42)
                                                       │
                       training split (5 634) ◄────────┴────────► test split (1 409) — untouched
                            │
        screen columns + infer schema (training split only)
                            │
   5-fold stratified CV, same folds for all 5 models, preprocessing re-fitted per fold
                            │
        select a model from CV scores only  ──► out-of-fold predictions of that model
                            │                              │
                            │                    choose the decision threshold
                            ▼
         refit on the whole training split ──► evaluate ONCE on the test split
```

* **No fitted step ever sees the test split.** Imputation statistics, scaling, one-hot categories, the column
  screen and the input schema are all fitted on training rows; during CV they are re-fitted inside each fold
  because they live in the scikit-learn `Pipeline`.
* **The test set does not select anything**: not the model, not hyperparameters (none are tuned), not the threshold.
  Only the selected model is evaluated on it. `tests/test_training.py::test_model_selection_ignores_test_labels…`
  flips every test label and asserts that CV scores, selection, out-of-fold predictions and threshold do not change.
* **Reproducible:** seeds for the split (`split.random_state`), the folds (`cv.random_state`) and every estimator are in the
  configuration; two runs give identical CV tables and test predictions (also tested). Wall-clock times differ.
  `python scripts/verify_reproducibility.py` re-runs everything and compares with the committed reports.

## 2. Models compared

| Key | Model | Why it is there |
|---|---|---|
| `dummy` | Predicts the class prior | The floor: anything that does not beat it learns nothing |
| `logistic_regression` | L2 logistic regression (`C=1`) | Interpretable reference; exact explanations |
| `decision_tree` | Depth ≤ 5, ≥ 20 samples per leaf | Simple readable rules; shows what one tree can do |
| `random_forest` | 300 trees, depth ≤ 12 | Bagged trees, strong default for tabular data |
| `gradient_boosting` | `HistGradientBoostingClassifier` | Captures interactions; no extra dependency |

Hyperparameters are fixed in the configuration and **not tuned** in this version: tuning on this small data set would
need nested cross-validation to stay honest, and was left out to keep the comparison simple and the claims modest.
No class weighting is used (see §4): probabilities stay interpretable as frequencies.

## 3. Choosing the model

* **Criterion:** mean CV **average precision** (configurable: ROC-AUC, F1, recall, precision, balanced accuracy,
  accuracy). Average precision summarises the precision–recall trade-off for the positive class; with 26.5 %
  positives and a campaign that can only contact some customers, how well the *top* of the ranking is purified matters
  more than accuracy (a model predicting "nobody churns" is 73.5 % accurate). ROC-AUC and the other metrics are
  always shown next to it.
* **Rule `one_se`** (default): among models whose mean is within one standard error of the best, take the *simplest*.
  A more complex model has to earn its place; a small gap inside the fold-to-fold noise is not evidence. `best` takes
  the highest mean. The dummy baseline is never selectable.
* **Result of the committed run** (identified by `run_id` in `models/metadata.json`): logistic regression, average precision 0.667 ± 0.020 (std over 5
  folds); random forest 0.661 ± 0.026, gradient boosting 0.664 ± 0.020, decision tree 0.611 ± 0.023, dummy 0.265.
  The top four are within noise of each other on this data; the simplest wins. See `reports/model_comparison.csv` and
  per-fold values in `reports/cv_folds.csv`.

### Metrics reported for every model and fold

ROC-AUC, average precision, precision, recall, F1, balanced accuracy, accuracy, log loss, Brier score, fit time and
score time (scikit-learn's per-fold prediction + scoring time, an upper bound on prediction time). In the CV table
precision, recall, F1, balanced accuracy and accuracy use the default 0.5 cut-off for every model; the recommended
threshold is applied only to the selected model's final evaluation.

## 4. Probabilities and calibration

The starting repository trained with `class_weight="balanced"`. On its held-out test set that model's mean predicted
churn probability was **0.413 while the observed churn rate was 0.265**: its scores inflated risk by about 55 %, so
any sum of probabilities (expected churners, expected benefit) would have been biased. ChurnLens trains without
class weights and checks calibration: on out-of-fold training predictions the selected model's mean predicted
probability is **0.2654 vs 0.2654 observed**, expected calibration error 0.011, Brier score 0.134. The risk segments
confirm it on the test set (Low 6.6 %, Medium 33.5 %, High 65.5 % observed churn for mean predicted 6.5 %, 34.1 %,
65.9 %). Calibration is measured on this data only and can drift on other populations.

## 5. Decision threshold

A threshold of 0.5 is not special. The recommended threshold is computed on **out-of-fold predictions of the training
split** by maximising F1 (alternatives: F2, Youden's J — `threshold.criterion`). For the committed run: **0.33**
(F1 0.640 on validation data; the curve is flat between 0.30 and 0.40). The app lets you move the threshold and
shows, for each value, how many customers are flagged and the false positives, false negatives, precision, recall and
campaign cost (`reports/threshold_analysis_validation.csv`). F1 is a generic statistical default: the economically
right threshold depends on contact cost and customer value — see
[ECONOMIC_ASSUMPTIONS.md](ECONOMIC_ASSUMPTIONS.md). Four things are always kept apart in the code and UI: the
**probability**, the **threshold**, the **predicted class** (probability ≥ threshold) and the **actual class**.

## 6. Final evaluation (test set, used once)

Reported for the selected model at the recommended threshold (and at 0.5 for reference), with 95 % bootstrap
intervals over test customers (`evaluation.bootstrap_intervals`). The intervals capture test-sample noise only: the
model is not refitted and nothing is said about a different population.

| Metric (threshold 0.33) | Value | 95 % interval |
|---|---|---|
| ROC-AUC | 0.842 | 0.823 – 0.865 |
| Average precision | 0.634 | 0.584 – 0.689 |
| Precision | 0.539 | 0.496 – 0.582 |
| Recall | 0.725 | 0.681 – 0.773 |
| F1 | 0.618 | 0.583 – 0.655 |

At threshold 0.5 the same model has precision 0.655, recall 0.532, F1 0.587. These describe *this* held-out sample of
the IBM sample data; they are not guarantees for any company's customers.

**Comparison with the starting repository.** On the same split, its saved model scored ROC-AUC 0.847 and average
precision 0.665 (recomputed during the audit). ChurnLens's 0.842 / 0.634 are slightly lower. Differences of this
size are within what a 1 409-row test set can produce (see the intervals), nothing was tuned towards the test set,
and the numbers are reported as they came out.

## 7. Feature engineering ablation

The four Telco-specific derived features (`TotalServices`, `ContractMonths`, `ChargeRatio`, `TenureSegment`) were
tested once, on CV only: logistic regression average precision **0.667 ± 0.020 with** vs **0.661 ± 0.019 without**
(ROC-AUC 0.848 vs 0.846). The gain is inside the noise. They are kept in the demo model because they are cheap,
interpretable and part of the reused code, but they are not claimed to help; `python train_model.py --no-engineer`
reproduces the other variant.

## 8. Column screening on user data

For a user's CSV (`data.infer_roles`, on the training split only): identifiers (by name or by unique integer values),
constant columns, date columns and very high-cardinality text are ignored; a column that **alone** separates the classes
(single-feature AUC ≥ 0.98, cross-fitted for text columns) is rejected as probable target leakage; ≥ 0.90 raises a
warning. Every decision is listed with its reason in the UI, and the user can knowingly keep a flagged column. The
target is never assumed to be the last column, and the positive class is never guessed silently.

## 9. Explainability

* **Global:** logistic-regression coefficients (log-odds per standard deviation / per category, with odds ratios);
  impurity importances for tree models (flagged as biased towards continuous features); **permutation importance**
  on the test set for any model, per raw input column.
* **Local:** for a linear model, an *exact* decomposition of the log-odds (intercept + contributions = decision function,
  checked to 1e-9 in the tests); for other models, one-at-a-time *baseline substitution* — the change in predicted
  probability when one feature is replaced by its training median / mode. The latter is not additive when features
  interact, and says so.
* Tests check the explanations against the model actually deployed (editing a coefficient changes the explanation as it
  should), not just that numbers come out.
* **SHAP was not added:** the explanations above cover the required cases without a heavy dependency, and
  compatibility with the pinned Python 3.14 / NumPy 2.4 stack was not verified.
* All explanations are **associations learned by the model**, not causes.

## 10. Known limitations of the method

* One train/test split and fixed hyperparameters; differences smaller than the CV standard deviation should not be
  over-interpreted.
* Snapshot data without dates: `tenure` and `TotalCharges` partly encode the outcome (see the data card); no temporal
  validation was possible.
* Probabilities were checked for calibration on validation data only.
* No fairness audit despite `gender` / `SeniorCitizen` being inputs.
* Models are pickles. They load only with the scikit-learn version that produced them, and only from a source you
  trust; the loader checks a SHA-256 recorded in `metadata.json` (integrity, not authenticity).
