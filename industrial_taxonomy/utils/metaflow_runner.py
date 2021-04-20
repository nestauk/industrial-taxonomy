import logging
import yaml
from itertools import chain
from pathlib import Path
from shlex import quote
from subprocess import Popen, CalledProcessError
from typing import List

import toolz.curried as t
from filelock import FileLock

from industrial_taxonomy import project_dir as PROJECT_DIR


logger = logging.getLogger(__file__)


def execute_flow(flow_file: Path, params: dict, metaflow_args: dict) -> int:
    """Execute flow in `flow_file` with `params`
    Args:
        flow_file (`pathlib.Path`): File containing metaflow
        params (`dict`): Keys are flow parameter names (command-line notation,
             `--`), values are parameter values (as strings).
    Returns:
        `int` - run_id of flow
    Raises:
        `CalledProcessError`
    """
    run_id_file = flow_file.parents[0] / ".run_id"
    cmd = " ".join(
        [
            "python",
            str(flow_file),
            "--no-pylint",
            *t.pipe(metaflow_args.items(), chain.from_iterable, t.map(quote)),
            "run",
            "--run-id-file",
            str(run_id_file),
            # PARAMS
            *t.pipe(params.items(), chain.from_iterable, t.map(quote)),
        ]
    )
    logger.info(cmd)

    # RUN FLOW
    proc = Popen(
        cmd,
        shell=True,
    )
    while proc.poll():
        print("poll")
        print(proc.communicate())
    proc.wait()
    return_value = proc.returncode

    if return_value != 0:
        raise CalledProcessError(return_value, cmd)
    else:
        with open(run_id_file, "r") as f:
            run_id = int(f.read())
        return run_id


def update_model_config(key_path: List[str], value: object) -> None:
    """Update subsection of `model_config.yaml`
    Reads and writes under a file-lock so that multiple completing flows
    do not over-write one another.
    Args:
        key_path (`list`): Path in dictionary to update
        value (`object`): Value to put in `key_path`
    """
    fname = PROJECT_DIR / "industrial_taxonomy" / "model_config.yaml"
    lock = FileLock(str(fname) + ".lock")
    with lock:
        # Read existing config
        with open(fname, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # Update
        config_ = t.assoc_in(config, key_path, value)
        # Write
        with open(fname, "w") as f:
            f.write(yaml.dump(config_, default_flow_style=False))
