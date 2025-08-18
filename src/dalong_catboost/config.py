from __future__ import annotations

import pathlib
from typing import Any, Dict

import yaml


def load_config(config_path: str | pathlib.Path) -> Dict[str, Any]:
    """Load a YAML config file into a nested dict.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        A dictionary with configuration sections such as data/model/fit.
    """
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping/dict")
    return cfg


