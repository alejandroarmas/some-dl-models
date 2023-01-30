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


setup:
	poetry run pre-commit install

# Styling
lint:
	poetry run black .
	poetry run flake8
	poetry run python -m isort .
	poetry run mypy .


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
