from pathlib import Path

from metaflow import Run, Flow

from industrial_taxonomy import config, project_dir
from industrial_taxonomy.utils.metaflow_runner import update_model_config, execute_flow


if __name__ == "__main__":
    FLOW_ID = 'sic4_embedder'
    preproc_run_id = config['flows']['sic4_classifier']['preproc']['run_id']
    
    embed_config = config["flows"][FLOW_ID]
    embed_flow_config = {
            '--model': embed_config['model'],
            '--preproc_flow_run_id': str(preproc_run_id)
            }
 
    flow_file = (Path(__file__).resolve().parents[0] / "embed.py")
    embed_run_id = execute_flow(
            flow_file,
            embed_flow_config,
            metaflow_args={}
            )
    embed_config["run_id"] = embed_run_id
    update_model_config(["flows", FLOW_ID], embed_config)

