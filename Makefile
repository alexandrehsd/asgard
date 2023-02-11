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
	sudo apt-get install liblapack-dev libblas-dev gfortran pkg-config libhdf5-dev

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
	pip install "pdm>=2.4.5"
	pdm install

.PHONY: requirements-lint
# install all requirements for code linting
requirements-lint:
	pdm install -G lint

.PHONY: requirements-test
# install all requirements for code testing
requirements-test:
	pdm install -G test

.PHONY: jupyter
# start a jupyter notebook server
jupyter:
	PYTHONPATH=$(shell pwd) python -m jupyter notebook

.PHONY: flake-check
## check PEP-8 style and other standards with flake8
flake-check:
	@echo ""
	@echo "\033[33mFlake 8 Standards\033[0m"
	@echo "\033[33m=================\033[0m"
	@echo ""
	@pdm run flake8 . \
	&& echo "\n\n\033[32mSuccess\033[0m\n" || (echo \
	"\n\n\033[31mFailure\033[0m\n\n\033[34mManually fix the offending \
	issues\033[0m\n" && exit 1)

.PHONY: black-check
## check Black code style
black-check:
	@echo ""
	@echo "\033[33mBlack Code Style\033[0m"
	@echo "\033[33m================\033[0m"
	@echo ""
	@pdm run black --check  . \
	&& echo "\n\n\033[32mSuccess\033[0m\n" || (pdm run black --diff . 2>&1 | grep -v -e reformatted -e done \
	&& echo "\n\033[31mFailure\033[0m\n\n\
	\033[34mRun \"\e[4mmake black\e[24m\" to apply style formatting to your code\
	\033[0m\n" && exit 1)

.PHONY: isort-check
## check PEP-8 and other standards with isort
# TODO: add folder 'tests' to check-only after create the folder
isort-check:
	@echo ""
	@echo "\033[33misort Standards\033[0m"
	@echo "\033[33m=================\033[0m"
	@echo ""
	@pdm run isort --check-only scripts asgard&& echo "\n\n\033[32mSuccess\033[0m\n" || (echo \
	"\n\n\033[31mFailure\033[0m\n\n\033[34mManually fix the offending \
	issues\033[0m\n" && exit 1)

.PHONY: checks
## perform code standards and style checks
checks: black-check flake-check isort-check

.PHONY: black
## apply the Black code style to code
black:
	@pdm run black .

.PHONY: isort
## apply the isort style to order imports
# TODO: add folder 'tests' to check-only after create the folder
isort:
	@pdm run isort scripts casio


