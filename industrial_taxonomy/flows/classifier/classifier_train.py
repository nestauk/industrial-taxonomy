"""Fine tune a transformer for classification using uniform batching and 
dynamic padding.
"""

import json
import logging
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from toolz.functools import pipe

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

class TextClassifier(FlowSpec):
#     documents_path = Parameter(
#             "documents_path",
#             help="Path to JSON training data",
#             type=str,
#             )
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
        self.preproc_run = Flow(
                f'{self.preproc_flow_class_name}/{preproc_run_id}').latest_run
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
        dataset = sort_by_char_len(self.eval_set)
        samples = [Sample(**x) for x in dataset]
        self.dataset = IterableDataset(samples, self.tokenizer, **self.config["encode"])
        self.next(self.join_datasets)

    @step
    def make_eval_set(self):
        dataset = sort_by_char_len(self.train_set)
        samples = [Sample(**x) for x in dataset]
        self.dataset = IterableDataset(samples, self.tokenizer, **self.config["encode"])
        self.dataset = _make_dataset(self.eval_set)

    @step
    def join_datasets(self, inputs):
        self.merge_artifacts(inputs, exclude=['datasets'])
        self.train_dataset = inputs.make_train_set.dataset
        self.eval_dataset = inputs.make_eval_set.dataset
        self.next(self.fine_tune)

    @step
    def fine_tune(self):
        model_config = self.config["model"]
        model_config["num_labels"] = self.preproc_run.data.n_labels
        logger.info('Loading pre-trained model')
        model = partial(model_init, model_config)
        training_args_config = self.config["training_args"]

        training_args = TrainingArguments(**training_args_config)
        trainer = Trainer(
                model=model(self.freeze_model),
                args=training_args,
                train_dataset=self.train_encodings,
                eval_dataset=self.eval_encodings,
                compute_metrics=compute_metrics,
                data_collator=BatchCollator(self.tokenizer)
                )
        logger.info(f'Training model with {len(self.train_encodings)} samples')
        trainer.train()
        self.model = trainer.model

        self.next(self.end)

    @step
    def end(self):
        pass

#     def _encode(self, dataset):
#         encode_config = self.config["encode"]
#         dataset = [Sample(**s) for s in dataset]
#         encodings = OrderedDataset(self.tokenizer, 
#                 dataset, encode_config, self.label_lookup)
#         return encodings
# 
#     @classmethod
#     def _load_trained_model(cls, run_id):
#         run = Run(f"{cls.__name__}/{run_id}")
#         return run.data.model
# 
#     @classmethod
#     def predict(cls, dataset, run_id, model):
#         run = Run(f"{cls.__name__}/{run_id}")
#         run_config = run.data.config
# 
#         samples = []
#         for sample in dataset:
#             if 'label' not in sample:
#                 sample['label'] = -1
#             samples.append(Sample(**sample))
# 
#         label_lookup = run.data.label_lookup
#         tokenizer = run.data.tokenizer
#         encodings = OrderedDataset(
#                 run.data.tokenizer,
#                 samples,
#                 run_config['encode'],
#                 run.data.label_lookup
#                 )
# 
#         inverse_label_lookup = {v: k for k, v in run.data.label_lookup.items()}
# 
#         trainer = Trainer(
#                 model=model,
#                 data_collator=BatchCollator(run.data.tokenizer)
#                 )
#         preds = trainer.predict(encodings)
#         pred_probs = softmax(preds.predictions, axis=1)
#         pred_labels = np.argmax(preds.predictions, axis=1)
#         pred_labels = [inverse_label_lookup[p] for p in pred_labels]
# 
#         doc_ids = [e.index for e in encodings.features]
# 
#         return pred_labels, pred_probs, doc_ids, inverse_label_lookup
#     @step
#     def generate_label_lookup(self):
#         """Maps sample labels to unique integer IDs and creates a lookup
#         """
#         self.label_lookup = dict()
#         i = 0
#         for doc in self.documents:
#             label = doc['label']
#             if label not in self.label_lookup:
#                 self.label_lookup[label] = i
#                 i += 1
# 
#         self.next(self.train_eval_split)
# 
#     @step
#     def train_eval_split(self):
#         """Splits the data into train and evaluation sets"""
#         split_config = self.config["train_eval_split"]
#         train_size = split_config.pop('train_size')
#         eval_size = split_config.pop('eval_size')
#         self.train_set, self.eval_set = train_test_split(self.documents, train_size=train_size,
#                 test_size=eval_size, **split_config)
#         self.next(self.encode_train_set, self.encode_eval_set)
# 
#     @step
#     def encode_train_set(self):
#         """Encodes the training dataset"""
#         self.encodings = self._encode(self.train_set)
#         self.next(self.encodings_join)
# 
#     @step
#     def encode_eval_set(self):
#         """Encodes the evaluation dataset"""
#         self.encodings = self._encode(self.eval_set)
#         self.next(self.encodings_join)
# 
#     @step
#     def encodings_join(self, inputs):
#         self.merge_artifacts(inputs, exclude=['encodings'])
#         self.train_encodings = inputs.encode_train_set.encodings
#         self.eval_encodings = inputs.encode_eval_set.encodings
#         self.next(self.fine_tune)

if __name__ == "__main__":
    TextClassifier()
