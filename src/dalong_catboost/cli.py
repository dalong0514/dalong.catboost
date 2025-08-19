from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Sequence

import pandas as pd

from .config import load_config
from .model import build_classifier, fit_classifier, save_model


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CatBoost classifier from a YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out", default="models/catboost.cbm", help="Where to save model")
    parser.add_argument(
        "--meta-out",
        default=None,
        help="Optional path to save training metadata (feature columns, categorical features). Defaults to <out>.meta.json",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    train_csv = data_cfg.get("train_csv")
    train_json = data_cfg.get("train_json")
    target = data_cfg.get("target")
    categorical = data_cfg.get("categorical", [])
    features: Sequence[str] | None = data_cfg.get("features")

    if not target:
        raise ValueError("data.target must be set in config")
    if not train_csv and not train_json:
        raise ValueError("One of data.train_csv or data.train_json must be set in config")

    if train_csv:
        df = pd.read_csv(train_csv)
    else:
        # train_json points to a JSON file that is either a list of objects or JSON Lines
        path = pathlib.Path(train_json)
        text = path.read_text(encoding="utf-8")
        text_stripped = text.lstrip()
        if text_stripped.startswith("["):
            records = json.loads(text)
            df = pd.DataFrame.from_records(records)
        else:
            # assume JSON Lines
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            df = pd.DataFrame.from_records(records)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not in training data")

    # Select features
    if features:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise KeyError(f"Configured features not found in data columns: {missing}")
        X = df[list(features)].copy()
    else:
        X = df.drop(columns=[target])

    y = df[target]

    # Auto-detect categorical features if not provided or explicitly set to 'auto'
    if not categorical or (isinstance(categorical, str) and categorical.lower() == "auto"):
        categorical_names = [c for c in X.columns if X[c].dtype == "object"]
    else:
        categorical_names = [c for c in categorical if c in X.columns]

    cat_indices = [X.columns.get_loc(c) for c in categorical_names]

    model_params = cfg.get("model", {})
    fit_params = cfg.get("fit", {})

    model = build_classifier(model_params)
    fit_classifier(model, X, y, cat_features=cat_indices, fit_parameters=fit_params)

    save_model(model, pathlib.Path(args.out))

    # Save training metadata for inference reproducibility
    meta_path = pathlib.Path(args.meta_out) if args.meta_out else pathlib.Path(str(args.out) + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "target": target,
        "features": list(X.columns),
        "categorical_names": list(categorical_names),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model saved to {args.out}")
    print(f"Metadata saved to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


