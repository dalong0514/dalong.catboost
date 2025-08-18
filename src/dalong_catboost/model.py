from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable, Optional

from catboost import CatBoostClassifier


def build_classifier(parameters: Dict[str, Any]) -> CatBoostClassifier:
    """Build a CatBoostClassifier from parameter mapping."""
    return CatBoostClassifier(**parameters)


def fit_classifier(
    model: CatBoostClassifier,
    X_train,
    y_train,
    cat_features: Optional[Iterable[int]] = None,
    fit_parameters: Optional[Dict[str, Any]] = None,
):
    """Fit the classifier with optional categorical feature indices and fit kwargs."""
    fit_kwargs: Dict[str, Any] = dict(fit_parameters or {})
    if cat_features is not None:
        fit_kwargs.setdefault("cat_features", list(cat_features))
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def save_model(model: CatBoostClassifier, output_path: str | pathlib.Path) -> None:
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(model_path: str | pathlib.Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


