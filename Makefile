.PHONY: install install-dev harness test cov lint typecheck fmt check clean

# The shared evaluation harness is a separate repo, not yet on PyPI.
# Prefer a sibling checkout so harness changes are picked up without a publish.
harness:
	@if [ -d ../eval-harness ]; then \
		echo "installing sibling ../eval-harness"; \
		pip install -e ../eval-harness; \
	else \
		echo "installing eval-harness from git"; \
		pip install "eval-harness @ git+https://github.com/mukundjaiswal/eval-harness.git"; \
	fi

install: harness
	pip install -e .

install-dev: harness
	pip install -e ".[dev]"
	pre-commit install

# Adds the heavy optional stack (train,scoring,dev). Not needed for tests.
install-full: harness
	pip install -e ".[train,scoring,dev]"

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing

lint:
	ruff check .

fmt:
	ruff format .
	ruff check --fix .

typecheck:
	mypy

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	find . -name __pycache__ -type d -exec rm -rf {} +
