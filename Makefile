# globals

# custom targets
.PHONY: environment
# setup python environment
environment:
	pyenv install -s 3.9.13 ;\
	pyenv virtualenv 3.9.13 sdg-classifier ;\
	pyenv local sdg-classifier

.PHONY: requirements
# install core requirements
requirements:
	pip install -Ur requirements.txt

.PHONY: requirements-lint
# install all requirements for code linting
requirements-lint:
	pip install -Ur requirements.lint.txt

.PHONY: jupyter
# start a jupyter notebook server
jupyter:
	PYTHONPATH=$(shell pwd) python -m jupyter notebook
