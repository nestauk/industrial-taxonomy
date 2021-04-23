import logging
import os
from pathlib import Path

from metaflow import namespace
import research_daps.flows.topsbm.topsbm as topsbm
from research_daps.flows.topsbm.sbmtm import sbmtm
Sbmtm = sbmtm

from industrial_taxonomy.utils.metaflow_client import flow_getter, cache_getter_fn
import industrial_taxonomy


logger = logging.getLogger(__name__)
namespace(None)


def get_topsbm_v1() -> Sbmtm:
    """TopSBM model trained on `getters.glass.get_description_tokens_v1` tokens.

    DEPRECATED!
    """
    logging.warning("THIS PIPELINE IS DEPRECATED! Use `get_description_tokens_v2`")

    cwd = os.getcwd()

    # Relies on `research_daps.flows.topsbm.Sbmtm` in local path
    os.chdir(Path(topsbm.__file__).parents[0])

    @cache_getter_fn
    def _get_topsbm_v1() -> Sbmtm:
        run_id = industrial_taxonomy.config["flows"]["topsbm_v1"]["run_id"]
        return flow_getter("TopSBMFlow", run_id=run_id).model

    model = _get_topsbm_v1()

    os.chdir(cwd)
    return model


def get_topsbm_v2() -> Sbmtm:
    """TopSBM model trained on `getters.glass.get_description_tokens_v2` tokens."""
    cwd = os.getcwd()

    # Relies on `research_daps.flows.topsbm.Sbmtm` in local path
    os.chdir(Path(topsbm.__file__).parents[0])

    @cache_getter_fn
    def _get_topsbm_v2() -> Sbmtm:
        run_id = industrial_taxonomy.config["flows"]["topsbm_v2"]["run_id"]
        return flow_getter("TopSBMFlow", run_id=run_id).model

    model = _get_topsbm_v2()

    os.chdir(cwd)
    return model
