import logging
from pathlib import Path

from industrial_taxonomy import config
from industrial_taxonomy.utils.metaflow_runner import update_model_config, execute_flow

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    FLOW_ID = "glass_desc_preproc"

    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]

    cmd_params = {
        "--n-gram": str(params["n_gram"]),
        "--test_mode": str(params["test_mode"]),
        "--n_docs": str(params["n_docs"]),
        "--max-workers": "1",
    }
    flow_file = Path(__file__).resolve().parents[0] / f"{FLOW_ID}.py"
    run_id = execute_flow(
        flow_file,
        cmd_params,
        metaflow_args={
            # "--datastore": "local",
            # "--metadata": "local",
        },
    )

    flow_config["run_id"] = run_id
    update_model_config(["flows", FLOW_ID], flow_config)
