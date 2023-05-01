.PHONY: install update test test-all lint format format-check clean clean-build build publish

PROJECT = minivan

install:
	poetry install

update:
	poetry update

clean:
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name ".coverage.*" -delete
	find . -type f -name "coverage.*" -delete

test: clean
	poetry run pytest --cov=$(PROJECT) --cov-report=xml --cov-report=term

format:
	poetry run black --config pyproject.toml .
	poetry run isort --profile black .

format-check:
	poetry run flake8 --config .flake8
	poetry run black --check  --config pyproject.toml .
	poetry run isort --check-only --diff --profile black .

clean-build:
	rm -rf dist
	rm -rf build

build: clean-build
	poetry build

publish:
	poetry publish
