# Notice and attributions

## Origin of the code

ChurnGuard AI started from the MIT-licensed repository
**[infabuilds/customer-churn-prediction](https://github.com/infabuilds/customer-churn-prediction)**
(copyright notice: *"Copyright (c) 2026 Customer Churn Prediction Project Contributors"*; the package
metadata of that repository names *Davide Infantino* as author). The MIT licence requires keeping that
notice, which is why [LICENSE](LICENSE) carries both copyright lines.

A second repository, [ROE10/customer-churn-prediction](https://github.com/ROE10/customer-churn-prediction),
was consulted only as inspiration for exploratory analysis; no code from it is included.

## What was kept, what is new

| | Kept from the original (idea or code) | New in ChurnGuard AI |
|---|---|---|
| Pipeline | One fitted scikit-learn `Pipeline` serialised once and reused for scoring (`ColumnTransformer`, `SimpleImputer`, `OneHotEncoder(handle_unknown="ignore")`); the Telco feature-engineering idea (service count, tenure segment, charge ratio) | Re-implemented in the `churnguard` package; engineered features corrected (explicit service list, neutral contract-length encoding, `TotalCharges = 0` when `tenure = 0`, open tenure bins); single input schema; data validation with readable errors |
| Models | Logistic regression and random forest | Five models on identical folds, dummy baseline, decision tree, gradient boosting; no class weighting (calibrated probabilities); fixed seeds |
| Evaluation | Selection by cross-validation on the training split only | Per-fold and mean ± std results for 9 metrics, configurable criterion with a one-standard-error rule, out-of-fold decision threshold, test set used once, calibration report, reproducible reports with run metadata |
| Explanations | Exact log-odds decomposition of the logistic model | Index-safe feature mapping, model-agnostic local explanation for non-linear models, global importances, permutation importance, tests that explanations match the deployed model |
| App | Streamlit prediction and exploration pages | Nine-section application, error analysis, threshold analysis, retention simulator, CSV export, guarded training on user data |
| API | FastAPI route layout (`/health`, `/predict`, `/predict/batch`) | Request model generated from the saved schema, integrity check of the model file, configurable CORS, tests |
| Not kept | PDF report generator with hard-coded figures; the "broken model" narrative | — |
| Tests / CI | none in the starting repository | Test suite, Ruff, GitHub Actions workflow |

## Data

The demonstration uses the **IBM "Telco Customer Churn" sample data set** (distributed through Kaggle as
`blastchar/telco-customer-churn`). Its licence could not be confirmed; the file is therefore **not
redistributed** by this project. See [data/README.md](data/README.md) and [docs/DATA_CARD.md](docs/DATA_CARD.md).

## Libraries

scikit-learn, pandas, NumPy, joblib, Plotly, Streamlit, FastAPI, Pydantic and Uvicorn are used under their
own open-source licences.
