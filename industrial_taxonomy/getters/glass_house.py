"""Data getter for Glass to Companies House matching."""
import logging

import pandas as pd
from metaflow import namespace

from industrial_taxonomy.utils.metaflow_client import cache_getter_fn


logger = logging.getLogger(__name__)
namespace(None)


@cache_getter_fn
def get_glass_house() -> pd.DataFrame:
    """Gets matches between Glass and Companies house (and accompanying score)."""
    # TODO:
    logger.warn("This Glass-House data is a temporary placeholder")
    s3_path = "s3://nesta-glass/data/processed/glass/company_numbers.csv"
    return pd.read_csv(s3_path)
