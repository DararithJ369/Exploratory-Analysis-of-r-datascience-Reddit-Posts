.PHONY: help install test run run-cached report jupyter docker-build docker-pipeline docker-jupyter docker-test clean

PYTHON ?= python3
UV ?= $(shell command -v uv 2>/dev/null)

help:
	@echo "Available commands:"
	@echo "  make install         Install project dependencies via uv or pip"
	@echo "  make test            Run unit test suite"
	@echo "  make run             Execute full analysis pipeline (with scraping fallback)"
	@echo "  make run-cached      Execute pipeline using cached dataset (no scraping)"
	@echo "  make report          Generate academic statistical markdown report"
	@echo "  make jupyter         Launch interactive JupyterLab on port 8888"
	@echo "  make docker-build    Build Docker container images"
	@echo "  make docker-pipeline Run pipeline inside Docker container"
	@echo "  make docker-jupyter  Start JupyterLab service inside Docker container"
	@echo "  make docker-test     Run unit test suite inside Docker container"
	@echo "  make clean           Remove temporary cache files and artifacts"

install:
ifdef UV
	uv pip install -r requirements.txt
else
	$(PYTHON) -m pip install -r requirements.txt
endif

test:
	$(PYTHON) -m unittest discover tests -v

run:
	$(PYTHON) main.py

run-cached:
	$(PYTHON) main.py --no-scrape

report:
	$(PYTHON) main.py --no-scrape --report data/statistical_report.md

jupyter:
	jupyter lab --port 8888

docker-build:
	docker compose build

docker-pipeline:
	docker compose up pipeline

docker-jupyter:
	docker compose up jupyter

docker-test:
	docker compose run --rm test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
