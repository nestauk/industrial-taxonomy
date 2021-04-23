import json
import logging
from pathlib import Path
from typing import Optional

from industrial_taxonomy import config
from industrial_taxonomy.getters.glass import get_organisation_description
from industrial_taxonomy.getters.glass_house import get_glass_house
from industrial_taxonomy.utils.metaflow_runner import update_model_config, execute_flow

logger = logging.getLogger(__name__)


def generate_input_data(match_threshold: int, n_docs: Optional[int] = None) -> Path:
    path = Path("input_data.json").resolve()
    if path.exists():
        return path
    else:
        (
            get_organisation_description()
            # Only orgs with a companies house match
            .merge(
                get_glass_house()
                .query(f"score > {match_threshold}")
                .drop(["company_number"], axis=1),
                on="org_id",
            )
            .sort_values("score", ascending=False)
            .drop(["score"], axis=1)
            .drop_duplicates(subset=["org_id"], keep="last")
            .set_index("org_id")["description"]
            .head(n_docs)  # head takes better matches (we are not random sampling)
            .to_json(path)
        )

    return path


if __name__ == "__main__":
    FLOW_ID = "nlp_flow"

    match_threshold = config["params"]["match_threshold"]
    flow_config = config["flows"][FLOW_ID]
    params = flow_config["params"]

    input_file = generate_input_data(match_threshold, params.get("n_docs", None))

    cmd_params = {
        "--n-gram": str(params["n_gram"]),
        "--entity-mappings": json.dumps(params["entity_mappings"]),
        "--input-file": str(input_file),
        "--n-process": str(params["n_process"]),
    }
    flow_file = Path(__file__).resolve().parents[0] / f"{FLOW_ID}.py"
    run_id = execute_flow(
        flow_file,
        cmd_params,
        metaflow_args={
            "--datastore": "local",
            "--metadata": "local",
            # "--with": "batch",
            "--environment": "conda",
        },
    )

    logging.info("UPDATING RUN ID")
    flow_config["run_id"] = run_id
    update_model_config(["flows", FLOW_ID], flow_config)
