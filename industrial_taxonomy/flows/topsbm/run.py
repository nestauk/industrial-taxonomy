"""Runs TopSBM flow on Glass descriptions."""
import json
from pathlib import Path
from uuid import uuid4

from toolz.curried import take, pipe
import research_daps.flows.topsbm.topsbm as topsbm

from industrial_taxonomy import config
from industrial_taxonomy.getters.glass import get_description_tokens
from industrial_taxonomy.utils.metaflow_runner import update_model_config, execute_flow


def generate_documents(n_docs: int) -> Path:
    """Generate notice tokens."""

    path = Path(f"{str(uuid4())[:8]}_topsbm.json").resolve()
    with open(path, "w") as f:
        pipe(
            get_description_tokens().items(),
            take(n_docs),
            dict,
            lambda data: json.dump(data, f),
        )

    return path


if __name__ == "__main__":

    flow_id = "topsbm"
    config_ = config["flows"][flow_id]
    params = config_["params"]

    filepath = generate_documents(params["n_docs"])

    cmd_params = {
        "--n-docs": str(params["n_docs"]),
        "--input-file": str(filepath),
    }
    flow_file = Path(topsbm.__file__).resolve()
    run_id = execute_flow(
        flow_file,
        cmd_params,
        metaflow_args={
            "--with": "batch:memory=64000,queue=job-queue-nesta-metaflow-test,image=metaflow-graph-tool",
            # "--datastore": "local",
            # "--metadata": "local",
        },
    )

    config_["run_id"] = run_id
    update_model_config(["flows", flow_id], config_)
