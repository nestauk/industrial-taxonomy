
import logging
import json
from pathlib import Path

from metaflow import Run

from industrial_taxonomy import config, project_dir
from industrial_taxonomy.utils.metaflow_runner import update_model_config, execute_flow
from industrial_taxonomy.flows.classifier.classifier_utils import create_org_data 

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    FLOW_ID = "sic4_classifier_no_nec"
    PREPROC_CLASS_NAME = "SicPreprocess"
    TRAIN_CLASS_NAME = "TextClassifier"

    ## PREPROCESSING ##
    preproc_config = config["flows"][FLOW_ID]["preproc"]
    preproc_flow_config = {
            '--match_threshold': str(preproc_config['match_threshold']),
            '--sic_level': str(preproc_config['sic_level']),
            '--test': str(preproc_config["test"]),
            '--nec_companies': str(preproc_config["nec_companies"]),
            '--config': json.dumps(preproc_config['config'])
            }

    flow_file = (Path(__file__).resolve().parents[0] / 
            f"{preproc_config['flow_id']}.py")
    preproc_run_id = execute_flow(
            flow_file,
            preproc_flow_config,
            metaflow_args={}
            )
    preproc_config["run_id"] = preproc_run_id
    update_model_config(["flows", FLOW_ID, "preproc"], preproc_config)

    ## TRAINING ##
    train_config = config["flows"][FLOW_ID]["train"]
    train_flow_config = {
            '--freeze_model': str(train_config['freeze_model']),
            '--preproc_run_id': str(preproc_run_id),
            '--preproc_flow_class_name': PREPROC_CLASS_NAME,
            '--config': json.dumps(train_config['config'])
            }

    flow_file = (Path(__file__).resolve().parents[0] / 
            f"{train_config['flow_id']}.py")
    train_run_id = execute_flow(
            flow_file,
            train_flow_config,
            metaflow_args={}
            )
    train_config["run_id"] = train_run_id
    update_model_config(["flows", FLOW_ID, "train"], train_config)

    ## PREDICTING ##
    predict_config = config["flows"][FLOW_ID]["predict"]
    predict_flow_config = {
            '--predict_proba': str(predict_config['predict_proba']),
            '--train_run_id': str(train_run_id),
            '--train_flow_class_name': TRAIN_CLASS_NAME
            }

    flow_file = (Path(__file__).resolve().parents[0] / 
            f"{predict_config['flow_id']}.py")
    predict_run_id = execute_flow(
            flow_file,
            predict_flow_config,
            metaflow_args={}
            )
    predict_config["run_id"] = predict_run_id
    update_model_config(["flows", FLOW_ID, "predict"], predict_config)
