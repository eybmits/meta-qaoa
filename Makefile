
.PHONY: setup lint test run

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -e .

lint:
	ruff check .
	black --check .

test:
	pytest

run:
	meta-qaoa --config configs/default.yaml
