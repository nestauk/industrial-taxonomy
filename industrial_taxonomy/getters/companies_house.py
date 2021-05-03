"""Data getters for Companies House data."""
import logging

import pandas as pd
from metaflow import namespace

from industrial_taxonomy.utils.metaflow_client import flow_getter, cache_getter_fn
import industrial_taxonomy


logger = logging.getLogger(__name__)
namespace(None)


def run_id() -> int:
    """Get `run_id` for flow

    NOTE: This is loaded from __init__.py not from file
    """
    return industrial_taxonomy.config["flows"]["companies_house"]["run_id"]


GETTER = flow_getter("CompaniesHouseMergeDumpFlow", run_id=run_id())


@cache_getter_fn
def get_organisation() -> pd.DataFrame:
    return GETTER.organisation


@cache_getter_fn
def get_address() -> pd.DataFrame:
    return GETTER.address


@cache_getter_fn
def get_sector() -> pd.DataFrame:
    """Returns most up-to-date sector rankings."""
    return (
        GETTER.organisationsector.sort_values("date")
        .drop_duplicates(["company_number", "rank"], keep="last")
        .rename(columns={"date": "data_dump_date", "sector_id": "SIC5_code"})
    )


@cache_getter_fn
def get_name() -> pd.DataFrame:
    return GETTER.organisationname
