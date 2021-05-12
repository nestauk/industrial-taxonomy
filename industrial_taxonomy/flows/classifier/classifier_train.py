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
       sort_by_char_len, Sample) 

logger = logging.getLogger(__name__)

class TextClassifier(FlowSpec):
    freeze_model = Parameter(
            "freeze_model",
            help="If True, layers before classification layer will be frozen",
            type=bool,
            default=False
            )
    preproc_run_id = Parameter(
            "preproc_run_id",
            help="Run ID of the preprocessing flow you are using",
            type=int,
            )
    preproc_flow_class_name = Parameter(
            "preproc_flow_class_name",
            help="Class name of the flow used for preprocessing",
            type=str,
            )
    config = Parameter(
            "config",
            help=("Config containing params for Trainer, TrainingArguments "
                  "and model"),
            type=JSONType,
            )

    @step
    def start(self):
        self.preproc_run = Run(
                f'{self.preproc_flow_class_name}/{self.preproc_run_id}')
        self.tokenizer = AutoTokenizer.from_pretrained(**self.config["tokenizer"])
        self.next(self.train_eval_split)

    @step
    def train_eval_split(self):
        """Splits the data into train and evaluation sets"""
        split_config = self.config["train_eval_split"]
        train_size = split_config.pop('train_size')
        eval_size = split_config.pop('eval_size')
        self.train_set, self.eval_set = train_test_split(
                self.preproc_run.data.train_set,
                train_size=train_size,
                test_size=eval_size, 
                **split_config)

        self.next(self.make_train_set, self.make_eval_set)

    @step
    def make_train_set(self):
        """Transforms the training samples into a transformers.Trainer compatible dataset."""
        dataset = sort_by_char_len(self.train_set)
        samples = [Sample(**x) for x in dataset]
        self.dataset = IterableDataset(samples, self.tokenizer, self.config["encode"])
        self.next(self.join_datasets)

    @step
    def make_eval_set(self):
        """Transforms the evaluation samples into a transformers.Trainer compatible dataset."""
        dataset = sort_by_char_len(self.train_set)
        samples = [Sample(**x) for x in dataset]
        self.dataset = IterableDataset(samples, self.tokenizer, self.config["encode"])
        self.next(self.join_datasets)

    @step
    def join_datasets(self, inputs):
        """Join artifacts from the training and evaluation dataset creation steps"""
        self.merge_artifacts(inputs, exclude=['dataset'])
        self.train_dataset = inputs.make_train_set.dataset
        self.eval_dataset = inputs.make_eval_set.dataset
        self.next(self.fine_tune)

    @step
    def fine_tune(self):
        """Instantiates a pre-trained model and fine tunes it for classification"""
        model_config = self.config["model"]
        model_config["num_labels"] = self.preproc_run.data.n_labels
        logger.info('Loading pre-trained model')
        model = partial(model_init, model_config)
        training_args_config = self.config["training_args"]

        training_args = TrainingArguments(**training_args_config)
        trainer = Trainer(
                model=model(self.freeze_model),
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=compute_metrics,
                data_collator=BatchCollator(self.tokenizer)
                )
        logger.info(f'Training model with {len(self.train_dataset)} samples')
        trainer.train()
        self.model = trainer.model

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    TextClassifier()
