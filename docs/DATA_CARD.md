# Data card — IBM Telco Customer Churn (demonstration data)

> Every figure below was computed from the file described in [data/README.md](../data/README.md)
> (SHA-256 `16320c9c…5e91`). Re-derive them with `python train_model.py` (see `models/metadata.json`) or in the
> app's *Data Explorer*.

## Source and licence

| | |
|---|---|
| Name | *Telco Customer Churn* (IBM sample data set) |
| Distribution | Kaggle `blastchar/telco-customer-churn`, file `WA_Fn-UseC_-_Telco_Customer_Churn.csv` |
| Licence | **Not verified.** A web search found no explicit licence on IBM's or Kaggle's pages; a Hugging Face mirror labels its copy CC-BY-4.0, but that is the mirror's claim, not the originator's. |
| Redistribution | **Not done by this project.** The file is not committed (`.gitignore`) and must be downloaded by each user under the terms shown on Kaggle (see `data/README.md`). |
| Nature of the records | The sources I could read do not say whether the rows are real customers or simulated. Treat the data as an illustrative sample, not as evidence about any real operator. |

## Content

One row per customer: **7 043 rows × 21 columns**, one customer per `customerID` (0 duplicated identifiers).

| Group | Columns | Type |
|---|---|---|
| Identifier | `customerID` | text — excluded from modelling, kept only to label exports |
| Demographics | `gender`, `SeniorCitizen` (0/1), `Partner`, `Dependents` | categorical / binary |
| Account | `tenure` (months, 0–72), `Contract`, `PaperlessBilling`, `PaymentMethod` | numeric / categorical |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | categorical (`No internet service` / `No phone service` are explicit levels) |
| Charges | `MonthlyCharges` (18.25–118.75), `TotalCharges` | numeric (`TotalCharges` is stored as text) |
| **Target** | `Churn` (`Yes`/`No`) | binary |

**Target distribution:** 1 869 `Yes` (26.54 %) and 5 174 `No`.

## Quality findings (all checked by code)

* **Missing values:** none as pandas `NaN`, but `TotalCharges` contains **11 blank strings**. All 11 rows have
  `tenure = 0` (no bill issued yet) and `Churn = No`. The pipeline sets `TotalCharges = 0` for `tenure = 0`; a blank on
  a row with `tenure > 0` would be median-imputed (and reported).
* **Duplicates:** 0 duplicated `customerID`; 22 rows are identical once the identifier is dropped. They are kept
  (distinct customers with identical attributes) and reported.
* **Constant columns:** none. **Identifier-like columns:** `customerID` only.
* **Consistency:** no row has `MultipleLines = "No phone service"` with `PhoneService = "Yes"`, and no row has a
  "No internet service" level with `InternetService != "No"`.
* **Outliers:** `MonthlyCharges` and `tenure` stay within plausible ranges (see above); no value was removed.
* **Redundancy:** `TotalCharges ≈ tenure × MonthlyCharges` (correlation 0.9996) — the three numeric columns carry
  overlapping information, which makes linear coefficients less interpretable.

## Descriptive associations (historical churn rates, not causes)

| Segment | Churn rate | Customers |
|---|---|---|
| Month-to-month contract | 42.7 % | 3 875 |
| One-year contract | 11.3 % | 1 473 |
| Two-year contract | 2.8 % | 1 695 |
| Fiber optic internet | 41.9 % | 3 096 |
| DSL | 19.0 % | 2 421 |
| No internet | 7.4 % | 1 526 |
| Electronic check | 45.3 % | 2 365 |
| Tenure 0–12 months | 47.4 % | — |
| Tenure 36–72 months | 11.9 % | — |

## Leakage and validity risks

* `tenure` and `TotalCharges` are **cumulative values observed when the file was extracted**. For a customer who
  churned they stop at departure, so part of their signal is a *consequence* of the outcome rather than a
  predictor available beforehand. With no dates in the file, a temporal split to measure this is impossible.
  **Status: documented hypothesis, not testable with this data.**
* Cleaning (trimming text, converting `TotalCharges`) is row-wise and happens before the split; every learned
  step (imputation statistics, scaling, encoding categories, column screening, schema) is fitted on the training
  split only.

## Sensitive attributes

`gender` and `SeniorCitizen` are present and are inputs of the demo model. No fairness audit was performed. Do
not use the scores for decisions about individuals without one.

## Processing applied by the project

1. Text trimmed; blank text → missing; numeric-looking text columns converted (`TotalCharges`).
2. Identifier separated from the features; target mapped to 1 = `Yes`.
3. Stratified 80/20 split (seed 42): 5 634 training rows, 1 409 held-out test rows.
4. Inside the model pipeline: `TotalCharges = 0` when `tenure = 0`, four derived features (`TotalServices`,
   `ContractMonths`, `ChargeRatio`, `TenureSegment`), median / most-frequent imputation, standardisation of numeric
   columns, one-hot encoding that ignores unseen categories.
