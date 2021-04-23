"""Utils to get and cache the results of metaflows."""
import logging
import os
import pickle
from pathlib import Path
from typing import Callable

from dotenv import find_dotenv, load_dotenv
from metaflow import Flow, Run
from metaflow.client.core import MetaflowData


logger = logging.getLogger(__file__)


def _get_temp_dir():
    """Find `temp_dir` env var or return None."""
    load_dotenv(find_dotenv())
    try:
        return os.environ["temp_dir"]
    except KeyError:
        return None


def cache_getter_fn(f: Callable):
    """Cache `f` output as pickle if `temp_dir` env var is set."""

    def inner(*args, **kwargs):
        temp_dir = _get_temp_dir()
        to_cache = True if temp_dir else False

        if not to_cache:
            return f(*args, **kwargs)
        else:
            cache_file = f.__qualname__
            cache_path = Path(temp_dir).resolve() / f.__module__
            cache_filepath = cache_path / cache_file

            if to_cache and not cache_path.exists():
                logger.info(f"Creating cache directory: {cache_path}")
                os.makedirs(cache_path, exist_ok=True)

            if cache_filepath.exists():
                logger.info(f"Loading from cache: {cache_filepath}")
                with open(cache_filepath, "rb") as fp:
                    return pickle.load(fp)
            else:
                logger.info(f"Caching: {cache_filepath}")
                out = f(*args, **kwargs)
                with open(cache_filepath, "wb") as fp:
                    pickle.dump(out, fp)
                return out

    return inner


def flow_getter(flow: str, run_id=None) -> MetaflowData:
    """Get data from a metaflow Run, checking it is valid.

    Args:
        flow (str): Metaflow flow name.
        run_id (int, optional): Metaflow run id. If None, get the latest successful run.

    Returns:
        Data from Run(flow/run_id).

    Raises:
        ValueError: If the run was not successful.
    """
    if run_id is None:
        run_id = Flow(flow).latest_successful_run.id
    run = Run(f"{flow}/{run_id}")
    if not run.successful:
        raise ValueError(f"Run({flow}/{run_id}).successful was False")
    return run.data
