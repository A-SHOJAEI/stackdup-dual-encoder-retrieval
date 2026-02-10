PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := /bin/bash
.DEFAULT_GOAL := all

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

CONFIG_BASELINE ?= configs/smoke_baseline.yaml
CONFIG_ABLATION ?= configs/smoke_no_hardneg.yaml

.PHONY: setup data train eval report all clean

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	@bash scripts/bootstrap_venv.sh

data: setup
	@$(PY) -m stackdup.data.synthetic --config configs/smoke_data.yaml

train: setup data
	@$(PY) -m stackdup.train.biencoder --config $(CONFIG_BASELINE)
	@$(PY) -m stackdup.train.biencoder --config $(CONFIG_ABLATION)

eval: setup data
	@$(PY) -m stackdup.baselines.tfidf_eval --config configs/smoke_data.yaml --out artifacts/tfidf_results.json
	@$(PY) -m stackdup.baselines.bm25_eval --config configs/smoke_data.yaml --out artifacts/bm25_results.json
	@$(PY) -m stackdup.eval.retrieval --config $(CONFIG_BASELINE) --out artifacts/biencoder_baseline_results.json
	@$(PY) -m stackdup.eval.retrieval --config $(CONFIG_ABLATION) --out artifacts/biencoder_ablation_results.json
	@$(PY) -m stackdup.reporting.collect_results \
		--inputs artifacts/tfidf_results.json artifacts/bm25_results.json artifacts/biencoder_baseline_results.json artifacts/biencoder_ablation_results.json \
		--out artifacts/results.json

report: setup
	@$(PY) -m stackdup.reporting.make_report --results artifacts/results.json --out artifacts/report.md

all: setup data train eval report

clean:
	@rm -rf $(VENV) artifacts runs .cache
