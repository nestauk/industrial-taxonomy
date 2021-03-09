"""Queries relating to sectors."""
import pandas as pd

from industrial_taxonomy import config
from industrial_taxonomy.getters.companies_house import get_sector
from industrial_taxonomy.getters.glass import get_organisation_description
from industrial_taxonomy.getters.glass_house import get_glass_house

MATCH_THRESHOLD = config["params"]["match_threshold"]


def get_glass_SIC_sectors(match_threshold: int = MATCH_THRESHOLD) -> pd.DataFrame:
    """SIC codes for glass organisations using Glass-Companies House matching.

    Args:
      match_threshold: Only keep matches with a score higher than this.
        Score can range from 0 to 100.
    """
    return (
        get_glass_house()
        .query(f"score > {match_threshold}")
        .drop("score", axis=1)
        .merge(
            get_sector(),
            on="company_number",
        )[["org_id", "SIC5_code", "rank", "data_dump_date"]]
    )


def get_glass_descriptions_SIC_sectors(
    match_threshold: int = MATCH_THRESHOLD,
) -> pd.DataFrame:
    """Combines sectors with descriptions

    Args:
      match_threshold: Only keep matches with a score higher than this.
        Score can range from 0 to 100.

    """

    sectors = get_glass_SIC_sectors()
    g_descr = get_organisation_description()

    sectors_descriptions = (
        sectors.merge(g_descr, on="org_id")
        # Add a division (SIC2) code
        .assign(division=lambda df: [x[:2] for x in df["SIC5_code"]])
        # Drop duplicated organisations
        .drop_duplicates(subset=["org_id"])
        .drop(axis=1, labels=["rank", "data_dump_date"])
        .reset_index(drop=True)
    )
    return sectors_descriptions
