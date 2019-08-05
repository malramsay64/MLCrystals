#
# Makefile
# Malcolm Ramsay, 2018-01-03 09:24
#
#
env_name = MLCrystals

.DEFAULT_GOAL = help

.PHONY: talks
talks: talks/2018-11-06-ICYRAM.pdf ## Compile talks

%.pdf: %.tex
	latexmk $< -output-directory=talks/output -xelatex
	cp talks/output/$(notdir $@) $@


all_notebooks = $(notdir $(wildcard notebooks/*.ipynb))

.PHONY: notebooks
notebooks: $(all_notebooks)

%.ipynb:
	cd notebooks && jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=None --execute $@

.PHONY: figures
figures:
	python3 src/figures.py labelled-config data/simulation/dataset/output/dump-Trimer-P1.00-T0.45-p2.gsd -i 2

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
