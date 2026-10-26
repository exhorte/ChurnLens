# ChurnLens

### From churn predictions to explainable, economically grounded retention decisions.

ChurnLens is an open-source customer churn analytics project that brings together a reproducible machine-learning workflow, an interactive Streamlit application, and a FastAPI scoring service. It is designed to help analysts investigate **who may churn, how reliable that assessment is, where a model makes mistakes, and what a retention strategy might look like under clearly stated assumptions**.

The project is an analytical and educational tool—not a production-ready decision system, a causal model, or a guarantee that a retention campaign will work.

> **Project naming:** This repository is presented as **ChurnLens**. The current application and API still use the inherited **ChurnGuard AI** label in some screens and responses, and the Python package is named `churnguard`.

## At a glance

- **Interactive app:** nine Streamlit sections for data exploration, model evaluation, customer risk, error analysis, explanations, retention scenarios, and exports.
- **Scoring API:** FastAPI endpoints for single-customer and batch predictions, using the same saved model and input schema as the app.
- **Evaluation discipline:** stratified train/test split, shared stratified cross-validation folds, out-of-fold threshold selection, and a held-out test set kept out of training, model selection, and threshold tuning.
- **Decision support:** adjustable probability threshold, risk segments, false-positive/false-negative analysis, and an assumption-driven retention simulator.
- **Reproducible artifacts:** a saved model, metadata, and CSV reports are included; the raw demonstration dataset is not redistributed.
- **Limitations made visible:** the demo is based on one historical sample dataset, has no temporal validation or fairness audit, and does not estimate campaign impact.

## What you can do

| App section | Purpose |
|---|---|
| **Dashboard** | Review the selected model, churn-risk distribution, and key evaluation context. |
| **Data Explorer** | Inspect and validate a dataset; upload a compatible CSV to train a session-only model. |
| **Model Comparison** | Compare five classifiers using cross-validation results and the separate final test evaluation. |
| **Customer Risk** | Score and rank customers, inspect probabilities and risk segments, and adjust the decision threshold. |
| **Error Analysis** | Examine false positives and false negatives when labelled customer data is available. |
| **Explainability** | Explore global feature importance and customer-level explanations. Explanations describe model associations, not causes. |
| **Retention Simulator** | Compare campaign scenarios using editable contact cost, customer value, assumed success rate, and contact constraints. |
| **Export** | Download scored customer results as CSV. |
| **About / Model Card** | Review the model, data, evaluation protocol, and known limitations. |

The saved demo model and its reports can load without the raw Telco CSV. Some customer-level views need the original dataset, however; without it, upload data or obtain the demo CSV as described in [data/README.md](data/README.md).

## How the machine-learning workflow works

The demo training protocol is intentionally designed to keep model selection and the final test evaluation separate:

1. Validate and prepare the labelled data; separate the target and any identifier from the model features.
2. Make a stratified 80/20 training/test split with a fixed seed.
3. Compare a Dummy Classifier, Logistic Regression, Decision Tree, Random Forest, and Histogram-based Gradient Boosting using the same five stratified folds on the training split.
4. Fit preprocessing within each model pipeline and each cross-validation fold.
5. Select a model using mean cross-validated **Average Precision** and the configured one-standard-error rule, which favours the simplest model within one standard error of the best.
6. Choose the decision threshold from the selected model’s out-of-fold training predictions (F1 is the current default).
7. Refit on the complete training split, then evaluate the selected model once on the held-out test set.

The test set does not select the model or threshold. It is used for final performance reporting and post-hoc permutation-importance analysis, not to tune the model. The complete protocol and its rationale are documented in [docs/METHODOLOGY.md](docs/METHODOLOGY.md); configuration is in [configs/config.toml](configs/config.toml).

### Demo model results

The committed run selected **Logistic Regression**. The following metrics are from the held-out test sample of the IBM Telco demonstration data—not a promise of performance on another company, time period, or customer population.

| Metric | At recommended threshold (0.33) | 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | 0.842 | 0.823–0.865 |
| Average Precision | 0.634 | 0.584–0.689 |
| Precision | 0.539 | 0.496–0.582 |
| Recall | 0.725 | 0.681–0.773 |
| F1 score | 0.618 | 0.583–0.655 |

The threshold of 0.33 was selected on out-of-fold training predictions by maximising F1; it was **not** selected on the test set. At the default 0.50 threshold, the same test sample has precision 0.655, recall 0.532, and F1 0.587. This illustrates the trade-off between flagging more potential churners and the precision of those flags.

Cross-validation results for the leading models are close: Logistic Regression has Average Precision **0.667 ± 0.020**, Gradient Boosting **0.664 ± 0.020**, and Random Forest **0.661 ± 0.026** (mean ± standard deviation across five folds). These differences should not be over-interpreted. Full model results, test metrics, and threshold reports are available under [reports/](reports/).

## Retention simulation: scenarios, not forecasts

The simulator compares strategies such as contacting nobody, contacting everyone, selecting the highest-risk customers, applying a probability threshold, or respecting a contact budget. It calculates expected churners targeted, campaign cost, gross and net benefit, ROI, and break-even values from the inputs provided.

Its core calculation uses each customer’s predicted churn probability and an assumed campaign success rate. **Churn prediction does not estimate the effect of an intervention.** The campaign success rate, customer value, and contact cost are user-supplied assumptions—not learned from the churn dataset. Results are therefore scenario calculations, not observed outcomes or recommendations. See [docs/ECONOMIC_ASSUMPTIONS.md](docs/ECONOMIC_ASSUMPTIONS.md) for formulas, example inputs, and limitations.

## Quick start

### Requirements

- Python **3.11 or newer**.
- The pinned dependencies in [requirements.txt](requirements.txt) for the app and training workflow.
- The optional Telco demonstration CSV if you want to retrain the demo model or use every dataset-dependent page.

Create an environment and install the app dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

### Run the Streamlit app

From the repository root:

```powershell
streamlit run app.py
```

The app uses the committed model and reports. If the demonstration CSV is absent, dataset-dependent customer views will be limited; you can upload a compatible dataset in the app or follow [data/README.md](data/README.md) to obtain the sample data.

### Obtain the demonstration data and retrain

The demonstration dataset is the IBM **Telco Customer Churn** sample distributed through Kaggle. Its original licence could not be verified, so the CSV is deliberately excluded from this repository. Read the source terms and download the file as described in [data/README.md](data/README.md), placing it at:

```text
data/WA_Fn-UseC_-_Telco_Customer_Churn.csv
```

Then run:

```powershell
python train_model.py
```

Training compares the configured models, evaluates the selected model, and writes the model and reports to `models/` and `reports/`. It will replace the corresponding local artifacts. To compare without the Telco-specific engineered features:

```powershell
python train_model.py --no-engineer
```

The script also accepts `--config`, `--data`, `--models-dir`, and `--reports-dir`; run `python train_model.py --help` for details. The data is not fabricated when missing, and the saved demo model does not remove the source dataset’s licensing or validation limitations.

## Scoring API

Install the API dependencies and start the development server:

```powershell
python -m pip install -r requirements-api.txt
uvicorn service.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

| Endpoint | Description |
|---|---|
| `GET /health` | Model availability and service health. |
| `GET /model` | Model metadata, selection, threshold, and evaluation context. |
| `GET /schema` | The saved model’s accepted input columns, types, and categories. |
| `POST /predict` | Score one customer. |
| `POST /predict/batch` | Score up to 1,000 customers in one request. |

The request contract is generated from the schema saved with the model. The API returns churn probability, predicted class, risk segment, applied threshold, and model label. The default CORS policy allows no cross-origin browser origins; configure `CHURNGUARD_CORS_ORIGINS` only for the origins you intend to permit. Set `CHURNGUARD_MODELS_DIR` to use a different model directory.

### Docker

The [Dockerfile](Dockerfile) builds a minimal API image and runs it as an unprivileged user:

```powershell
docker build -t churnlens-api .
docker run --rm -p 8000:8000 churnlens-api
```

The container includes the model artifacts present at build time. Use only model files from a trusted source: scikit-learn pickle files can execute code when loaded. The SHA-256 check detects an integrity mismatch against the metadata; it does **not** establish that a model is authentic or safe.

## Project structure

```text
.
├── app.py                       Streamlit entry point
├── service/                     FastAPI scoring service
├── src/churnguard/              Data, schema, pipeline, training, scoring, and analysis
├── ui/                          Streamlit pages and shared UI logic
├── configs/config.toml          Reproducible modelling and simulation settings
├── models/                      Saved model and metadata
├── reports/                     Cross-validation, test, threshold, and explanation reports
├── data/                        Dataset instructions; raw sample data is not included
├── docs/                        Methodology, data card, and economic assumptions
├── notebooks/                   Exploratory analysis and model-training notebooks
├── tests/                        Unit, integration, API, and UI tests
├── train_model.py               Demo-model training command
└── scripts/verify_reproducibility.py
```

## Tests, lint, and reproducibility

Install development dependencies and run the test suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
ruff check .
```

The tests use small synthetic fixtures where appropriate; synthetic test data is not used to claim real-world model performance. Some UI tests require the saved model artifacts, and dataset-specific tests are skipped when the unredistributed Telco CSV is unavailable.

To rerun the complete demo training protocol and compare its outputs with the committed reports and metadata, download the matching CSV first, then run:

```powershell
python scripts/verify_reproducibility.py
```

The verification script does not overwrite the committed artifacts.

## Data, responsible use, and limitations

- **Demonstration data:** 7,043 rows and 21 columns from the IBM Telco Customer Churn sample distributed through Kaggle. The data file is not included because its original redistribution licence could not be verified; see [docs/DATA_CARD.md](docs/DATA_CARD.md).
- **Historical snapshot:** the data has no dates. `tenure` and `TotalCharges` may partly reflect events observed at or after a customer’s departure. Temporal validation is not possible with this sample.
- **No causal claims:** a high feature importance or local explanation indicates an association used by the model, not a cause of churn or evidence that changing a feature will prevent it.
- **No campaign-effect model:** the simulator’s intervention success rate and financial inputs are assumptions. The model does not determine which customers are persuadable.
- **No fairness audit:** the sample includes gender and senior-citizen attributes, which are model inputs. Fairness has not been assessed.
- **Population shift:** calibration and test metrics describe this sample and split. Validate the data, calibration, operating threshold, and outcomes on the population and time period where the model would be used.
- **Human oversight:** do not use these predictions as the sole basis for decisions about individuals. Review applicable privacy, fairness, consumer-protection, and data-governance requirements before any real-world use.
- **Model artifact safety:** only load pickle files from trusted sources and compatible dependency versions.

Read [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and [docs/DATA_CARD.md](docs/DATA_CARD.md) before interpreting the results.

## Roadmap

The longer-term vision is to evolve ChurnLens from a churn-scoring demonstration into a trustworthy retention decision-support platform. The following are **future directions, not claims about functionality already delivered**:

1. Validate on consented, representative, time-stamped customer data and add temporal evaluation.
2. Audit fairness, calibration, and model performance across relevant customer groups and over time.
3. Evaluate retention interventions using experimental or otherwise credible outcome data before making claims about campaign effectiveness.
4. Strengthen deployment, monitoring, data governance, and operational safeguards before considering production use.

## Credits and licence

ChurnLens builds on the MIT-licensed [infabuilds/customer-churn-prediction](https://github.com/infabuilds/customer-churn-prediction) project. The repository’s [NOTICE.md](NOTICE.md) documents the origin of the code, what was adapted, and the contributions made here. The MIT licence and required copyright notices are included in [LICENSE](LICENSE).

The IBM Telco sample data is separate from the software licence and is not redistributed by this project. Review its source terms before downloading or using it.

## Documentation

- [Methodology and evaluation results](docs/METHODOLOGY.md)
- [Demonstration dataset instructions](data/README.md)
- [Dataset card and data risks](docs/DATA_CARD.md)
- [Retention simulation assumptions](docs/ECONOMIC_ASSUMPTIONS.md)
- [Code origin and attributions](NOTICE.md)
