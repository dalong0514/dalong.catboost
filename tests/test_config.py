from __future__ import annotations

import io
import tempfile
from pathlib import Path

import yaml

from dalong_catboost.config import load_config


def test_load_config_roundtrip(tmp_path: Path) -> None:
    cfg = {"a": 1, "b": {"c": 2}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    loaded = load_config(path)
    assert loaded == cfg


