# Makefile
SHELL = /bin/bash


.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "lint   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."
	@echo "test    : runs the entire unittest suite."
	@echo "setup    : installs all dependencies."
	@echo "stage_2   : trains our stage 2 MLP."


stage_2:
	poetry run python -m script.stage_2_script.stage_2_script_mlp
	poetry run python -m script.stage_2_script.stage_2_load_result

setup:
	poetry run pre-commit install

# Styling
lint:
	poetry run black .
	poetry run flake8
	poetry run python -m isort .
	poetry run mypy . --ignore-missing-imports


test:
	poetry run python -m unittest discover

# Environment
venv:
	poetry shell

# Cleaning
.PHONY: clean
clean: lint
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage
