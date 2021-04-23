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

## Generating reports

- Place [pandoc markdown](https://pandoc.org/MANUAL.html#pandocs-markdown) report files in `output/` - e.g. `output/<your files>.md`
- Place figures in `output/figures/`
- Place bibtex entries in `output/bibliography.bib`
- Export altair charts to both PNG (stored and comitted locally) and JSON spec (uploaded to s3) with `industrial_taxonomy.utils.altair_s3.export_chart()`
  ```python
  chart: altair.Chart = ...

  export_chart(chart, "filepath/to/figure")
  # Locally saves PNG: `industrial_taxonomy/output/figures/filepath/to/figure.png"
  # Uploads JSON spec to: `s3://industrial-taxonomy/figures/filepath/to/figure.json"
  ```
- To embed an interactive altair figure (that has been exported with the above step) in the HTML output use the following syntax: `![my figure caption](figures/path/to/png_version_of_image.png){#fig:my-fig-ref .altair}`.
  The `.altair` tells the pandoc filter that runs in the conversion stage to substitute the image for the JSON spec which is passed to [Vega-Embed](https://github.com/vega/vega-embed) to make it interactive.
- Generate a HTML/PDF report with `bin/report_it.py`, e.g. `./bin/report_it.py html input.md --publish` Converts `input.md` to `input.html` and publishes it (and `output/figures`) to S3.

  For more options: `./bin/report_it.py --help` & `./bin/report_it.py <subcommand> --help`

  Running this command requires your `industrial-taxonomy` conda environment to be activated.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/nestauk/cookiecutter-data-science-nesta">Nesta cookiecutter data science project template</a>.</small></p>
