from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Sequence

import pandas as pd

from .model import load_model


def _load_records_from_path(path: pathlib.Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    # JSON lines
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _build_dataframe(records: List[dict], features: Sequence[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=list(features))
    df = pd.DataFrame.from_records(records)
    # Add missing columns as None
    for col in features:
        if col not in df.columns:
            df[col] = None
    # Keep only configured features and preserve order
    df = df[list(features)]
    return df


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained CatBoost model on JSON input")
    parser.add_argument("--model", required=True, help="Path to .cbm model file")
    parser.add_argument("--meta", required=False, help="Path to training metadata JSON (<model>.meta.json by default)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="A single JSON object as string for one record")
    group.add_argument("--input-file", help="Path to a JSON file (array of objects or JSON Lines)")
    parser.add_argument("--proba", action="store_true", help="Output class probabilities as well")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    model_path = pathlib.Path(args.model)
    meta_path = pathlib.Path(args.meta) if args.meta else pathlib.Path(str(model_path) + ".meta.json")

    model = load_model(model_path)

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}. Re-train or specify --meta.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    features: Sequence[str] = meta.get("features") or []

    if args.input:
        record = json.loads(args.input)
        records = [record]
    else:
        records = _load_records_from_path(pathlib.Path(args.input_file))

    X = _build_dataframe(records, features)

    preds = model.predict(X)
    # Convert numpy arrays to Python lists for JSON serializable output
    if args.proba:
        proba = model.predict_proba(X)
        # CatBoost returns list-of-lists for multi-class; ensure native types
        output = [
            {"prediction": (pred if not hasattr(pred, "item") else pred.item()), "proba": list(map(float, p))}
            for pred, p in zip(preds, proba)
        ]
    else:
        output = [pred if not hasattr(pred, "item") else pred.item() for pred in preds]

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


