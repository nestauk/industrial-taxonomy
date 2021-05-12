"""Runs TopSBM flow on Glass descriptions."""
import json
from pathlib import Path
from uuid import uuid4

from research_daps.flows.topsbm import topsbm

from industrial_taxonomy import config
from industrial_taxonomy.getters.glass import get_description_tokens_v2
from industrial_taxonomy.utils.metaflow_runner import update_model_config, execute_flow


def generate_documents() -> Path:
    """Generate notice tokens."""

    path = Path(f"{uuid4()}.json").resolve()
    with open(path, "w") as f:
        json.dump(get_description_tokens_v2(), f)

    return path


if __name__ == "__main__":

    flow_id = "topsbm"
    config_ = config["flows"][flow_id]
    params = config_["params"]

    filepath = generate_documents()

    cmd_params = {
        "--n-docs": str(params["n_docs"]),
        "--input-file": str(filepath),
    }
    flow_file = Path(topsbm.__file__).resolve()
    run_id = execute_flow(
        flow_file,
        cmd_params,
        metaflow_args={
            "--with": "batch:memory=64000,image=metaflow-graph-tool",
        },
    )

    config_["run_id"] = run_id
    update_model_config(["flows", flow_id], config_)

