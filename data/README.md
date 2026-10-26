# Data

This folder is intentionally empty in the repository: the demonstration data set is **not redistributed**
because its licence could not be verified (see [docs/DATA_CARD.md](../docs/DATA_CARD.md)).

## Getting the demo data

The project expects the IBM *Telco Customer Churn* sample data set, as published on Kaggle:

1. Open <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> and read its terms.
2. Download the archive (needs a Kaggle account) and extract `WA_Fn-UseC_-_Telco_Customer_Churn.csv`.
3. Put the file here: `data/WA_Fn-UseC_-_Telco_Customer_Churn.csv`.

Or, with a configured Kaggle API token (`~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d blastchar/telco-customer-churn -p data --unzip
```

## Checking you have the same file

The reported results were produced from a file with these properties (also stored in
`models/metadata.json`):

| | |
|---|---|
| Rows × columns | 7 043 × 21 |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |

```bash
python -c "import hashlib,sys; print(hashlib.sha256(open('data/WA_Fn-UseC_-_Telco_Customer_Churn.csv','rb').read()).hexdigest())"
```

A different version of the file trains a different model: the app detects the mismatch and disables the
customer-level pages until you retrain (`python train_model.py`).

## Other data

* **Your own CSV** can be uploaded in the app (*Data Explorer*) to train a model, or in the sidebar to score new
  customers. Uploaded data stays in the session and is never written to disk by the app.
* **The tests never need this file**: they use small synthetic data generated in `tests/conftest.py`, clearly
  labelled as synthetic and never used to claim anything about real performance.
* The project never fabricates data in place of the real file: a missing file is an error with instructions.
