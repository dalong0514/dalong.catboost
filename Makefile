PY := python3
PIP := $(PY) -m pip
VENV := .venv

.PHONY: venv install format lint test train-example kernel lab nbstrip-install nbstrip nb

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

kernel:
	. $(VENV)/bin/activate; python -m ipykernel install --user --name dalong-catboost --display-name "Python (dalong-catboost)"

lab:
	. $(VENV)/bin/activate; jupyter lab

nb:
	. $(VENV)/bin/activate; jupyter notebook

nbstrip-install:
	. $(VENV)/bin/activate; nbstripout --install --attributes .gitattributes

nbstrip:
	. $(VENV)/bin/activate; nbstripout notebooks/**/*.ipynb || true


