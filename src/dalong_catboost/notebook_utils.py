from __future__ import annotations

import pathlib
import sys
from typing import Optional


def find_repo_root(start_path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Find repository root by looking for a pyproject.toml upwards."""
    current = (start_path or pathlib.Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def ensure_repo_in_sys_path(start_path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Ensure `<repo>/src` is importable in notebooks.

    Returns the resolved repository root path.
    """
    root = find_repo_root(start_path)
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return root


