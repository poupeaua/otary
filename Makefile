#!/bin/bash

.PHONY: docs

# ------------------------------------- CLEAN -----------------------------------------

clean: clean-global clean-cache

clean-global:
	@echo "\n---------------"
	@echo "Cleaning Global"
	@echo "---------------"
	# docs
	rm -r docs/build
	rm -r docs/source/api

	# coverage
	rm .coverage
	rm -r htmlcov junit

clean-cache:
	@echo "\n---------------"
	@echo "Cleaning Caches"
	@echo "---------------"
	rm -r .mypy_cache .pytest_cache .ruff_cache
	find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

# -------------------------------------- INIT -----------------------------------------

init: venv-check init-install

init-install:
	@echo "\n-----------------------"
	@echo "Installing Dependencies"
	@echo "-----------------------"

	@echo -e Upgrading pip...
	@python -m pip install --upgrade pip
	@echo -e Done

	@echo -e Installing Python Dependencies
	@poetry install
	@echo -e Done

# -------------------------------------- TEST -----------------------------------------

test: test-unit

test-unit:
	@echo "\n------------"
	@echo "Unit Testing"
	@echo "------------"
	python -m pytest --doctest-modules \
		--junitxml=junit/test-unit-results.xml \
		--cov=otary --cov-report=xml --cov-report=html tests/unit/

check: check-pylint check-ruff check-mypy check-black

check-pylint:
	@echo "\n------------------------------"
	@echo "Checking code quality - Pylint"
	@echo "------------------------------"
	@pylint otary/ --fail-under=9.5

check-ruff:
	@echo "\n----------------------------"
	@echo "Checking code quality - Ruff"
	@echo "----------------------------"
	@ruff check otary/

check-mypy:
	@echo "\n----------------------------"
	@echo "Checking code quality - Mypy"
	@echo "----------------------------"
	@mypy otary/

check-black:
	@echo "\n----------------------------"
	@echo "Checking code quality - Black"
	@echo "----------------------------"
	@black otary/ --check

# -------------------------------------- DOCS -----------------------------------------

docs-serve:
	@echo "\n------------------"
	@echo "Serve documentation"
	@echo "-------------------"
	poetry run mkdocs serve

docs-deploy:
	@echo "\n-------------------"
	@echo "Deploy documentation"
	@echo "--------------------"
	poetry run mkdocs gh-deploy

# -------------------------------------- VENV ------------------------------------------

venv-check:
	@echo "\n----------------------------"
	@echo "Checking Virtual Environment"
	@echo "----------------------------"

	@if [ ${VIRTUAL_ENV} != "" ]; then\
		echo "You are in a python virtual environment (${VIRTUAL_ENV}).";\
	else\
		echo "Please activate a python virtual environment.";\
		exit 1;\
		echo "Verify not printed";\
	fi
