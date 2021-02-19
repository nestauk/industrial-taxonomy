"""Data getters for Glass business website data."""
import pandas as pd
from metaflow import namespace

import industrial_taxonomy
from industrial_taxonomy.utils.metaflow_client import flow_getter, cache_getter_fn

namespace(None)


def run_id() -> int:
    """Get `run_id` for flow

    NOTE: This is loaded from __init__.py not from file
    """
    return industrial_taxonomy.config["flows"]["glass"]["run_id"]


GETTER = flow_getter("GlassMergeMainDumpFlow", run_id=run_id())


@cache_getter_fn
def get_organisation() -> pd.DataFrame:
    """Glass organisations."""
    return GETTER.organisation


@cache_getter_fn
def get_address() -> pd.DataFrame:
    """Address information extracted from Glass websites (longitudinal)."""
    return GETTER.organisationaddress.merge(GETTER.address, on="address_id").drop(
        "address_id", 1
    )


@cache_getter_fn
def get_sector() -> pd.DataFrame:
    """Sector (LinkedIn taxonomy) information for Glass Businesses (longitudinal)."""
    return GETTER.organisationsector.merge(GETTER.sector, on="sector_id").drop(
        "sector_id", 1
    )


@cache_getter_fn
def get_organisation_description() -> pd.DataFrame:
    """Description of business activities for Glass businesses (longitudinal)."""
    return GETTER.organisationdescription


@cache_getter_fn
def get_organisation_metadata() -> pd.DataFrame:
    """Metadata for Glass businesses (longitudinal)."""
    return GETTER.organisationmetadata