industrial-taxonomy
==============================

ESCoE project to explore shortcomings of SIC taxonomy and build a new industrial taxonomy using business website data.

## Setup

Ensure you have AWS configured, and [git-crypt](https://github.com/AGWA/git-crypt) installed.

To configure git, build the environment, and setup metaflow simply run `make install`.

### Cache metaflow getters

Adding `temp_dir` in a `.env` file, e.g. `temp_dir=/Users/<username>/GIT/industrial-taxonomy/data/interim/` will cause the metaflow getters to be pickled and cached in `temp_dir`. This cache will be checked and used first when running a getter - this saves downloading large files each time you run a getter at the expense of using more disk-space.

Note: metaflow does do some limited caching by itself (i.e. without setting `temp_dir`) but is not as consistent and will not persist between reboots.

## Functionality

### Experimental keyword-extraction and keyword community detection.

Run `python industrial_taxonomy/scripts/extract_description_keywords.py` to extract keywords from a sample of test sectors.

Run `python industrial_taxonomy/scripts/make_test_taxonomy.py` to build a network of keyword co-occurrences and extract keyword communities indicative of sectors.

Control the sectors used in these experiments in `industrial_taxonomy/model_config.yaml`.

## Code-style

Please run `make lint` to format your code to a common style, and to lint code with flake8.

## Approach to notebooks

Jupyter notebooks are great for exploration and presentation but cause problems for working collaboratively.

Use [Jupytext](https://jupytext.readthedocs.io/en/latest/) to automatically convert notebooks to and from `.py` format, commit the `.py` version (`.ipynb` files are ignored by git).

This allows us to separate code from output data, facilitating easier re-factoring, testing, execution, and code review.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/nestauk/cookiecutter-data-science-nesta">Nesta cookiecutter data science project template</a>.</small></p>
