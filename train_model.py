"""Train, compare and persist the ChurnGuard demo model on the Telco dataset.

    python train_model.py                      # defaults from configs/config.toml
    python train_model.py --no-engineer        # without the Telco feature engineering
    python train_model.py --config my.toml --models-dir /tmp/m --reports-dir /tmp/r

Writes (see docs/METHODOLOGY.md): models/churn_pipeline.pkl, models/metadata.json and
reports/{model_comparison,cv_folds,oof_predictions,test_predictions,test_evaluation}.csv
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from churnguard.config import ConfigError, load_config
from churnguard.data import (
    DatasetNotFoundError,
    DataValidationError,
    load_demo_dataset,
    prepare_dataset,
    read_csv,
    verify_demo_dataset,
)
from churnguard.training import run_experiment, save_experiment

logger = logging.getLogger("train_model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=Path, help="TOML configuration (default: configs/config.toml)")
    parser.add_argument("--data", type=Path, help="Telco CSV (default: path from the configuration)")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--no-engineer", action="store_true", help="Disable Telco feature engineering")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = load_config(args.config)
        data_file = args.data or config.data_path
        if args.data:
            raw = read_csv(args.data)
            verify_demo_dataset(raw, config)
        else:
            raw = load_demo_dataset(config)
        prepared = prepare_dataset(
            raw,
            config.data.target,
            positive_label=config.data.positive_label,
            id_column=config.data.id_column,
            config=config,
        )
        for note in prepared.notes:
            logger.info("data: %s", note)
        result = run_experiment(
            prepared,
            config,
            engineer=not args.no_engineer,
            data_file=data_file,
            progress=logger.info,
        )
        paths = save_experiment(
            result, args.models_dir or config.models_dir, args.reports_dir or config.reports_dir
        )
    except (ConfigError, DataValidationError, DatasetNotFoundError) as exc:
        logger.error("Training aborted: %s", exc)
        return 1

    cols = ["label", "roc_auc_mean", "roc_auc_std", "average_precision_mean",
            "average_precision_std", "f1_mean", "log_loss_mean", "brier_score_mean", "fit_time_s_mean"]
    print("\nCross-validation on the training split (mean over folds):")
    print(result.comparison[cols].round(4).to_string(index=False))
    print(f"\nSelected : {result.selected_label}\n  {result.selection_note}")
    print(f"Threshold: {result.threshold.threshold:.2f}  ({result.threshold.description})")
    print("\nFinal test set (used once, selected model only):")
    print(pd.Series(result.test_metrics).round(4).to_string())
    print("\nArtefacts:")
    for name, path in paths.items():
        print(f"  {name:17s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
