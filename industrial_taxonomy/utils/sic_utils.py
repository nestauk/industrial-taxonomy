# Create sic4 code - name lookup

import requests
import re
import pandas as pd
import industrial_taxonomy
import os
import logging
from typing import Dict

from toolz import keymap

project_dir = industrial_taxonomy.project_dir

_SIC_OUTPUT_FILE = f"{project_dir}/data/raw/sic_2007.xls"
_SIC_TAXONOMY_URL = "https://www.ons.gov.uk/file?uri=/methodology/classificationsandstandards/ukstandardindustrialclassificationofeconomicactivities/uksic2007/sic2007summaryofstructurtcm6.xls"


def save_sic_taxonomy():
    """Fetch SIC taxonomy and save as excel file"""
    if os.path.exists(_SIC_OUTPUT_FILE) is False:
        logging.info("Fetching SIC taxonomy")
        response = requests.get(_SIC_TAXONOMY_URL)
        with open(_SIC_OUTPUT_FILE, "wb") as f:
            f.write(response.content)
    else:
        logging.info("Already fetched SIC taxonomy")


def load_sic_taxonomy():
    """Load SIC taxonomy into a dataframe"""
    return pd.read_excel(
        _SIC_OUTPUT_FILE,
        skiprows=1,
        dtype={"Division": str, "Group": str, "Class": str, "Sub Class": str},
    )


def section_code_lookup() -> Dict[str, str]:
    """Returns lookup from 2-digit SIC code to SIC section letter."""

    def _dictrange(key_range, value) -> dict:
        return {i: value for i in key_range}

    return keymap(
        lambda i: str(i).zfill(2),
        {
            **_dictrange([1, 2, 3], "A"),
            **_dictrange(range(5, 10), "B"),
            **_dictrange(range(10, 34), "C"),
            35: "D",
            **_dictrange(range(36, 40), "E"),
            **_dictrange(range(41, 44), "F"),
            **_dictrange(range(45, 48), "G"),
            **_dictrange(range(49, 54), "H"),
            **_dictrange([55, 56], "I"),
            **_dictrange(range(58, 64), "J"),
            **_dictrange([64, 65, 66], "K"),
            68: "L",
            **_dictrange(range(69, 76), "M"),
            **_dictrange(range(77, 83), "N"),
            84: "O",
            85: "P",
            **_dictrange([86, 87, 88], "Q"),
            **_dictrange(range(90, 94), "R"),
            **_dictrange([94, 95, 96], "S"),
            **_dictrange([97, 98], "T"),
            99: "U",
        },
    )


def extract_sic_code_description(var_name):
    """Extracts codes and descriptions from SIC table
    Args:
        var_name (str): level of SIC we want to extract a lookup for
    Returns:
        A lookup between the variable codes and their description
    """
    table = load_sic_taxonomy()

    loc = list(table.columns).index(var_name)  # Find the location of class
    select = table.iloc[:, [loc, loc + 1]].dropna()  # Extract variable & description
    select.columns = [var_name, "description"]

    select[var_name] = [
        re.sub(r"\.", "", str(x).strip()).strip() for x in select[var_name]
    ]
    return select.set_index(var_name)["description"].to_dict()
