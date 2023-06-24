# globals
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = asgard
PYTHON_VERSION_FULL = $(shell python --version 2>&1)
PIP_VERSION_FULL = $(wordlist 1,2,$(shell pip --version 2>&1))

.PHONY: install-dependencies
# install system dependencies needed to run the environment packages
install-dependencies:
	@echo "Installing Dependencies"
	sudo apt-get update
	sudo apt-get install liblapack-dev libblas-dev gfortran pkg-config libhdf5-dev \
	build-essential libssl-dev zlib1g-dev \
	libbz2-dev libreadline-dev libsqlite3-dev curl \
	libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# custom targets
.PHONY: environment
# setup python environment
environment:
	pyenv install -s 3.9.13 ;\
	pyenv virtualenv 3.9.13 asgard ;\
	pyenv local asgard

.PHONY: requirements
# install core requirements
requirements:
	pip install --upgrade pip
	pip install pdm
	pdm install

.PHONY: requirements-lint
# install all requirements for code linting
requirements-lint:
	pdm install -G lint

.PHONY: requirements-test
# install all requirements for code testing
requirements-test:
	pdm install -G test

.PHONY: datasets
# pull datasets from dvc google-drive remote storage
datasets:
	@pdm run dvc pull --remote gdrive-remote-asgard-datasets

.PHONY: jupyter
# start a jupyter notebook server
jupyter:
	PYTHONPATH=$(shell pwd) python -m jupyter notebook

.PHONY: code-check
## perform code standards and style checks
code-check:
	@pdm run ruff check --format=github .

.PHONY: tests
tests:
	@pdm run pytest
