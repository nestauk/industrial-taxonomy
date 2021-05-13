#!/bin/env python
"""Typer CLI to generate HTML / pdf reports."""
import logging
from pathlib import Path
from typing import List, Optional

import sh
import typer

from industrial_taxonomy import project_dir

BUCKET = "industrial-taxonomy"
app = typer.Typer()
state = {"verbose": False}


@app.callback()
def callback(verbose: bool = False):
    """Convert pandoc markdown to a snazzy report.

    Convert to HTML with html command.

    Convert to PDF (via. LaTeX) with the pdf command.
    """  # noqa: DAR101
    if verbose:
        state["verbose"] = True
    else:
        logging.disable()


@app.command()
def pdf(
    inputs: List[Path] = typer.Argument(
        ..., help="One or more markdown files to concatenate and convert"
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Output path. Defaults to first value of INPUTS... with a PDF file suffix",
    ),
):
    """Convert markdown INPUTS...  to pdf OUTPUT."""
    _validate_inputs(inputs)

    if output is None:
        output = Path(inputs[0]).with_suffix(".tex")

    # TODO: fix s.t. figure paths work if run outside `output/`
    metadata_file = Path(project_dir) / "output/latex_metadata.yaml"
    _pandoc(inputs, output, metadata_file, variable="urlcolor=blue", citeproc=True)

    typer.secho(f"Report output at file://{output.resolve()}", bg="green", fg="black")


@app.command()
def html(
    inputs: List[Path] = typer.Argument(
        ..., help="One or more markdown files to concatenate and convert"
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Output path. Defaults to the first value of INPUTS... with a HTML file suffix",  # noqa: B950
    ),
    publish: bool = typer.Option(False, help="Publish document to s3"),
):
    """Convert markdown INPUTS...  to html OUTPUT."""
    _validate_inputs(inputs)

    if output is None:
        output = Path(inputs[0]).with_suffix(".html")

    metadata_file = Path(project_dir) / "output/html_metadata.yaml"
    _pandoc(inputs, output, metadata_file, self_contained=True, toc=True,
            citeproc=True, mathml=True)

    typer.secho(f"Report output at file://{output.resolve()}", bg="green", fg="black")

    if publish:
        sh.aws.s3.cp(
            Path(project_dir) / "output/figures/",
            f"s3://{BUCKET}/figures",
            recursive=True,
            acl="public-read",
        )
        key = output.resolve().relative_to(Path(project_dir))
        sh.aws.s3.cp(
            output,
            f"s3://{BUCKET}/{key}",
            acl="public-read",
        )
        typer.secho(
            f"Published report to https://{BUCKET}.s3.amazonaws.com/{key}",
            bg="green",
            fg="black",
        )


def _pandoc(inputs, output, metadata_file, **kwargs):
    """Run pandoc to convert `inputs` to `output`."""
    return sh.pandoc(
        *map(str, inputs),
        metadata_file=metadata_file,
        o=output,
        f="markdown",
        F="pandoc-crossref",  # natbib needs to go before
        filter=Path(project_dir) / "bin/altair_pandoc_filter.py",
        metadata=f"bucket={BUCKET}",
        number_sections=True,
        bibliography=Path(project_dir) / "output/bibliography.bib",
        **kwargs,
    )


def _validate_inputs(inputs: List[Path]) -> None:
    """Exit if not all `inputs` have markdown file suffix."""
    if not all(file.suffix == ".md" for file in inputs):
        typer.secho("ERROR: All inputs must be markdown", bg="red", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
