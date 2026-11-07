PY ?= python

.PHONY: setup test lint experiment spy

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

experiment:
	$(PY) scripts/run_regimes.py

spy:   # uses data/SPY.csv if present, else downloads (pip install -e ".[real-data]")
	$(PY) scripts/run_regimes.py --ticker SPY $(if $(wildcard data/SPY.csv),--csv data/SPY.csv,)
