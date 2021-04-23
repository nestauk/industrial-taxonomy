"""Data getters for SIC taxonomy."""
from typing import Dict

from metaflow import namespace

import industrial_taxonomy
from industrial_taxonomy.utils.metaflow_client import flow_getter, cache_getter_fn

namespace(None)

RUN_ID = industrial_taxonomy.config["flows"]["sic"]["run_id"]

_getter = flow_getter("Sic2007Structure", RUN_ID)


@cache_getter_fn
def division_lookup() -> Dict[str, str]:
    """SIC division name lookup."""
    return _getter.division


def level_lookup(level: int) -> Dict[str, str]:
    """Get SIC names for `level`.

    Args:
        level: Number of SIC digits/letters to fetch lookup for

    Returns:
        Lookup from SIC code/letter to name
    """
    levels = ["section", "division", "group", "class_", "subclass"]

    if level not in range(1, len(levels) + 1):
        raise ValueError(f"Level: {level} not valid.")

    return getattr(_getter, levels[level - 1])
