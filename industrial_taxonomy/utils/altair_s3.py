"""Altair S3 export utilities."""
import os
import tempfile
from mimetypes import guess_type
from pathlib import Path
from uuid import uuid4

import boto3

from industrial_taxonomy import project_dir


def alt_to_s3(chart, bucket, key):
    """Save altair chart json to S3

    Args:
        chart (altair.vegalite.v4.api.Chart): Altair chart object to save
        bucket (str): Name of s3 bucket to save chart in
        key (str): Object key (i.e. path) within bucket to save chart to
    """
    s3 = boto3.client("s3")

    suffix = Path(key).suffix or "json"
    content_type = guess_type(key)[0] or "text/json"

    fname = f"{tempfile.gettempdir()}/{str(uuid4())}.{suffix}"
    chart.save(fname)
    with open(fname, "rb") as f:
        # Upload html, giving public read permissions,
        #  and with html content type metadata
        s3.upload_fileobj(
            f,
            bucket,
            key,
            ExtraArgs={"ContentType": content_type, "ACL": "public-read"},
        )
    os.remove(fname)  # Cleanup temporary file


def export_chart(chart, key, bucket="industrial-taxonomy", static_alt_chart=None):
    """Export Altair `chart` to S3 as spec and locally as png.

    S3 goes to s3://`bucket`/`key`; local goes to
     `project_dir`/figures/`key`.
    """
    if static_alt_chart is None:
        static_alt_chart = chart

    alt_to_s3(chart, bucket, f"figures/{key}.json")

    path = Path(f"{project_dir}/output/figures/{key}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    static_alt_chart.save(str(path), format="png")
