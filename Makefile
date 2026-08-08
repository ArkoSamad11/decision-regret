# Every pipeline stage is its own target; `reproduce` chains the ones this
# pass implements. See docs/DECISIONS.md for what M5-M9 (transfer study,
# DeepSets encoder, support gate, counterfactuals) still need.

PYTHON := api/.venv/Scripts/python.exe
CONFIG ?= base.yaml

.PHONY: install ingest features train calibrate report serve-db reproduce serve web test lint

install:
	cd api && python -m venv .venv || py -3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e "api[dev]"
	cd web && npm ci

ingest:
	$(PYTHON) -m xdr.data.ingest --config $(CONFIG)

features: ingest
	$(PYTHON) -m xdr.features.build --config $(CONFIG)

train: features
	$(PYTHON) -m xdr.models.train --config $(CONFIG)

calibrate: train
	$(PYTHON) -m xdr.models.calibrate --config $(CONFIG)

report: calibrate
	$(PYTHON) -m xdr.evaluation.report --config $(CONFIG) --split test

serve-db: calibrate
	$(PYTHON) -m xdr.serve.store --config $(CONFIG)

# ingest -> features -> train -> calibrate -> report -> serve-db.
# Transfer/ablation/support-gate stages join this chain in later milestones.
reproduce: report serve-db

serve:
	$(PYTHON) -m uvicorn xdr.serve.app:app --host 0.0.0.0 --port 8000

web:
	cd web && npm run dev

test:
	cd api && .venv/Scripts/python.exe -m pytest tests -v

lint:
	$(PYTHON) -m ruff check api/src api/tests
