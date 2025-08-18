from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List

import pandas as pd

from .config import load_config
from .model import build_classifier, fit_classifier, save_model


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CatBoost classifier from a YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out", default="models/catboost.cbm", help="Where to save model")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    train_csv = data_cfg.get("train_csv")
    target = data_cfg.get("target")
    categorical = data_cfg.get("categorical", [])
    if not train_csv or not target:
        raise ValueError("data.train_csv and data.target must be set in config")

    df = pd.read_csv(train_csv)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not in training data")

    X = df.drop(columns=[target])
    y = df[target]
    cat_indices = [X.columns.get_loc(c) for c in categorical if c in X.columns]

    model_params = cfg.get("model", {})
    fit_params = cfg.get("fit", {})

    model = build_classifier(model_params)
    fit_classifier(model, X, y, cat_features=cat_indices, fit_parameters=fit_params)

    save_model(model, pathlib.Path(args.out))
    print(f"Model saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


