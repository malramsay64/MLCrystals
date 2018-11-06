#
# Makefile
# Malcolm Ramsay, 2018-01-03 09:24
#
#
lockfile = environment-lock.txt
env_name = MLCrystals

.DEFAULT_GOAL = help

jupyter:
	docker run --rm --port 8888:8888

build:  ## Build docker image
	docker build .

.PHONY: talks
talks: talks/2018-11-06-ICYRAM.pdf ## Compile talks

%.pdf: %.tex
	latexmk $< -output-directory=talks/output -xelatex
	cp talks/output/$(notdir $@) $@

.PHONY: data
data: ## Download datasets
	curl -L "https://drive.google.com/uc?export=download&id=1EslO883MNH_1ebQrKkjozKbLGakbKzr2" -o data/dataset.tar.xz
	mkdir -p data/simulation/trimer
	tar xvJf data/dataset.tar.xz -C data/simulation/trimer

lock: | ${lockfile}  ## Create or update the exact dependencies to install
	docker run --rm\
		--volume $(shell pwd):/srv:z \
		--workdir /srv \
		continuumio/miniconda3:4.5.4 bash -c \
		"conda env create -f environment.yml && source activate ${env_name} && conda list --explicit > ${lockfile}"

# The file needs to exist to be able to mount it to a docker container
${lockfile}:
	touch $@

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
