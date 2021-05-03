"""Fine tune a transformer for classification using uniform batching and 
dynamic padding.
"""

import json
import logging
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path

from metaflow import FlowSpec, step, Parameter, JSONType, S3, Run
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer)

from industrial_taxonomy.flows.classifier.classifier_utils import (
       BatchCollator, IterableDataset, compute_metrics, model_init,
       sort_by_char_len) 

logger = logging.getLogger(__name__)

class TextClassifierPredict(FlowSpec):
    train_run_id = Parameter(
            "train_run_id",
            help="Run ID of the model training flow you are using",
            type=int,
            )
    train_flow_class_name = Parameter(
            "train_flow_class_name",
            help="Class name of the flow used for training",
            type=str,
            )
    predict_proba = Parameter(
            "predict_proba",
            help="If True, also returns prediction probabilities",
            type=bool,
            default=True
            )

    def _make_dataset(self, dataset):
        tokenizer_config = self.train_run.data.config["tokenizer"]
        encode_config = self.train_run.data.config["encode"]
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
        samples = [Sample(**x) for x in dataset]
        return IterableDataset(samples, tokenizer, encode_config)

    @step
    def start(self):
        self.train_run = Flow(
                f'{self.train_flow_class_name}/{train_run_id}').latest_run
        self.next(self.make_test_set)

    @step
    def make_test_set(self):
        test_set = self.train_run.data.preproc_run.data.test_set
        dataset = sort_by_char_len(test_set)
        samples = [Sample(**x) for x in dataset]
        self.test_dataset = IterableDataset(
                samples, 
                self.tokenizer, 
                **train_run.data.config["encode"])
        self.next(self.predict)

    @step 
    def predict(self):
        trainer = Trainer(
                model=self.train_run.data.model,
                data_collator=BatchCollator(self.train_run.data.tokenizer)
                )

        preds = trainer.predict(encodings)
        self.pred_labels = np.argmax(preds.predictions, axis=1)
        if self.predict_proba:
            self.pred_probs = softmax(preds.predictions, axis=1)
        
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    TextClassifierPredict()
