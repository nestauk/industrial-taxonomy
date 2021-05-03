from sentence_transformers import SentenceTransformer

from metaflow import FlowSpec, step, Parameter, Run
from sentence_transformers import SentenceTransformer

from industrial_taxonomy.flows.classifier.classifier_utils import (
       sort_by_char_len) 


class Embedder(FlowSpec):
    model = Parameter(
            "model",
            help="Model version to use",
            type=str,
            )
    preproc_run_id = Parameter(
            "preproc_flow_run_id",
            help="Run ID of the preprocessing flow",
            type=int
            )
    @step
    def start(self):
        self.preproc_run = Run(
                f'SicPreprocess/{self.preproc_run_id}')
        self.next(self.embed)
    
    @step
    def embed(self):
        model = SentenceTransformer(self.model)
        texts = (t['text'] for t in self.preproc_run.data.org_data)

        self.encodings = model.encode(texts)

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    Embedder()

