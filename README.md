industrial-taxonomy
==============================

ESCoE project to explore shortcomings of SIC taxonomy and build a new industrial taxonomy using business website data.

## Setup

To configure git and build the environment simply run `make install`.

## Code-style

Please run `make lint` to format your code to a common style, and to lint code with flake8.

## Approach to notebooks

Jupyter notebooks are great for exploration and presentation but cause problems for working collaboratively.

Use [Jupytext](https://jupytext.readthedocs.io/en/latest/) to automatically convert notebooks to and from `.py` format, commit the `.py` version (`.ipynb` files are ignored by git).

This allows us to separate code from output data, facilitating easier re-factoring, testing, execution, and code review.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/nestauk/cookiecutter-data-science-nesta">Nesta cookiecutter data science project template</a>.</small></p>
