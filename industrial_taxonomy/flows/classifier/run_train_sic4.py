import logging
import json
from pathlib import Path

from industrial_taxonomy import config, project_dir
from industrial_taxonomy.utils.metaflow_runner import update_model_config, execute_flow
from industrial_taxonomy.flows.classifier.classifier_utils import create_org_data 

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    FLOW_ID = "classifier_train"
    CONFIG_ID = "sic4_classifier"
    task_config = config["flows"][CONFIG_ID]
    output_dir = task_config["config"]["training_args"]["output_dir"]
    task_config["config"]["training_args"]["output_dir"] = str(project_dir / output_dir)
    flow_config = {
            '--documents_path': str(project_dir / task_config['documents_path']),
            '--freeze_model': str(task_config['freeze_model']),
            '--config': json.dumps(task_config['config'])
            }

    flow_file = Path(__file__).resolve().parents[0] / f"{FLOW_ID}.py"
    run_id = execute_flow(
            flow_file,
            flow_config,
            metaflow_args={}
            )

    flow_config["run_id"] = run_id
    update_model_config(["flows", CONFIG_ID], task_config)
