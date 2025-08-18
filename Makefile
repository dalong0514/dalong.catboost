PY := python3
PIP := $(PY) -m pip
VENV := .venv

.PHONY: venv install format lint test train-example

venv:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate; $(PIP) install -U pip setuptools wheel

install: venv
	. $(VENV)/bin/activate; $(PIP) install -e .[dev]

format:
	. $(VENV)/bin/activate; ruff check --fix .
	. $(VENV)/bin/activate; black .
	. $(VENV)/bin/activate; isort .

lint:
	. $(VENV)/bin/activate; ruff check .
	. $(VENV)/bin/activate; mypy src || true

test:
	. $(VENV)/bin/activate; pytest -q

train-example:
	. $(VENV)/bin/activate; dalong-catboost --config configs/example_binary.yaml --out models/catboost.cbm


