"""Re-run the whole training protocol and compare it with the committed artefacts.

    python scripts/verify_reproducibility.py

Exit code 0 if the cross-validation table, the selected model, the threshold and the test
predictions are identical (up to 1e-9) to ``reports/`` and ``models/``; 1 otherwise. Needs the
Telco CSV (see data/README.md) and takes about a minute. Nothing in the repository is overwritten.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from churnguard.config import load_config
from churnguard.data import DatasetNotFoundError, load_demo_dataset, prepare_dataset
from churnguard.predictions import load_bundle
from churnguard.training import run_experiment, save_experiment

TIME_COLUMNS = ("fit_time_s", "score_time_s")


def main() -> int:
    config = load_config()
    try:
        raw = load_demo_dataset(config)
    except DatasetNotFoundError as exc:
        print(f"Cannot verify: {exc}")
        return 1
    prepared = prepare_dataset(raw, config.data.target, positive_label=config.data.positive_label,
                               id_column=config.data.id_column, config=config)
    result = run_experiment(prepared, config, engineer=True, data_file=config.data_path)
    problems: list[str] = []

    committed = pd.read_csv(config.reports_dir / "model_comparison.csv").set_index("model")
    fresh = result.comparison.set_index("model")
    numeric = [c for c in fresh.columns if fresh[c].dtype.kind == "f" and not c.startswith(TIME_COLUMNS)]
    for column in numeric:
        if not np.allclose(fresh[column], committed.loc[fresh.index, column], atol=1e-9, equal_nan=True):
            problems.append(f"model_comparison.csv differs in column '{column}'")

    saved = load_bundle(config.models_dir)
    if saved.metadata["model_key"] != result.selected_key:
        problems.append(f"selected model differs: {saved.metadata['model_key']} vs {result.selected_key}")
    if abs(saved.threshold - result.threshold.threshold) > 1e-12:
        problems.append(f"threshold differs: {saved.threshold} vs {result.threshold.threshold}")
    committed_test = pd.read_csv(config.reports_dir / "test_predictions.csv")
    if not np.allclose(committed_test["proba"], result.test_predictions["proba"], atol=1e-9):
        problems.append("test_predictions.csv differs")

    with tempfile.TemporaryDirectory() as tmp:
        save_experiment(result, Path(tmp) / "models", Path(tmp) / "reports")

    if problems:
        print("NOT reproducible:")
        for item in problems:
            print(" -", item)
        return 1
    print(f"Reproducible: {len(numeric)} CV columns x {len(fresh)} models, selection '{result.selected_key}', "
          f"threshold {result.threshold.threshold:.2f} and {len(committed_test)} test predictions all match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
