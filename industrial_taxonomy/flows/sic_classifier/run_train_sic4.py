import logging
import json
from pathlib import Path

from industrial_taxonomy import config
from industrial_taxonomy.utils.metaflow_runner import update_model_config, execute_flow
from industrial_taxonomy.flows.sic_classifier.classifier_utils import create_org_data 

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    FLOW_ID = "classifier_train"
    task_config = config["flows"][FLOW_ID]
    flow_config = task_config.pop("flow")
    flow_config = {
            '--documents_path': flow_config['documents_path'],
            '--freeze_model': str(flow_config['freeze_model']),
#             '--output_dir': flow_config['output_dir'],
            '--config': json.dumps(flow_config['config'])
            }
    print(flow_config)

    flow_file = Path(__file__).resolve().parents[0] / f"{FLOW_ID}.py"

    # data = create_org_data(**task_config)
    run_id = execute_flow(
            flow_file,
            flow_config,
            metaflow_args={}
            )

    flow_config["run_id"] = run_id
    update_model_config(["flows", FLOW_ID], flow_config)
