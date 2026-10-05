PY ?= python3
.PHONY: all test sims example
all: test sims example
test:
	$(PY) -m unittest discover -s tests -v
sims:
	$(PY) simulations/run_simulations.py
example:
	$(PY) examples/checkout_experiment.py
