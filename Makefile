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

spy:   # needs: pip install -e ".[real-data]" and internet access
	$(PY) scripts/run_regimes.py --ticker SPY
