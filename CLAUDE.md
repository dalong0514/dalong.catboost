# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A CatBoost machine learning project scaffold with configuration-driven training and CLI tools for both training and inference. Supports binary and multiclass classification with automatic categorical feature handling.

## Key Commands

**Development Environment:**
```bash
make install      # Create venv and install dependencies
make format       # Format code with ruff, black, isort
make lint         # Run linting checks
make test         # Run tests
```

**Training:**
```bash
dalong-catboost --config configs/example_binary.yaml --out models/model.cbm
```

**Inference:**
```bash
dalong-catboost-predict --model models/model.cbm --input '{"feature1": 1.0, "feature2": "value"}'
```

**Example Training:**
```bash
make train-example  # Train with example config
```

## Architecture

**Core Components:**
- `src/dalong_catboost/cli.py` - Main training CLI
- `src/dalong_catboost/predict_cli.py` - Inference CLI  
- `src/dalong_catboost/config.py` - YAML config loader
- `src/dalong_catboost/model.py` - CatBoost model utilities

**Configuration Structure:**
- YAML configs in `configs/` directory
- Three main sections: `data`, `model`, `fit`
- Supports CSV and JSON/JSONL input formats
- Automatic categorical feature detection

**Data Flow:**
1. Load config → 2. Read training data → 3. Build model → 4. Train → 5. Save model + metadata

**Inference:** Uses `.meta.json` files for feature schema reproducibility

## Configuration Examples

**Binary Classification (`configs/example_binary.yaml`):**
```yaml
data:
  train_csv: data/train.csv
  target: label
  categorical: []

model:
  loss_function: Logloss
  iterations: 200
  learning_rate: 0.1
```

**Multiclass (`configs/instrument_multiclass.yaml`):**
```yaml
data:
  train_csv: data/instrument.csv
  target: label
  features: [function, instrument_type, ...]
  categorical: auto

model:
  loss_function: MultiClass
  iterations: 800
  learning_rate: 0.06
```

## Development Notes

- Python >= 3.9 required
- Uses modern packaging with pyproject.toml
- Testing with pytest
- Code formatting: black, isort, ruff
- Type checking: mypy (optional)
- Jupyter notebook support available