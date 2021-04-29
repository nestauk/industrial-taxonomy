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
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer)

from industrial_taxonomy.flows.sic_classifier.classifier_utils import (
       BatchCollator, OrderedDataset, compute_metrics, model_init, Sample) 

logger = logging.getLogger(__name__)

class TrainTextClassifier(FlowSpec):
    documents_path = Parameter(
            "documents_path",
            help="Path to JSON training data",
            type=str,
            )
    freeze_model = Parameter(
            "freeze_model",
            help="If True, layers before classification layer will be frozen",
            type=bool,
            default=False
            )
    config = Parameter(
            "config",
            help=("Config containing params for Trainer, TrainingArguments "
                  "and model"),
            type=JSONType,
            )

    def _encode(self, dataset):
        encode_config = self.config["encode"]
        dataset = [Sample(**s) for s in dataset]
        encodings = OrderedDataset(self.tokenizer, 
                dataset, encode_config, self.label_lookup)
        return encodings

    @classmethod
    def load_trained_model(cls, run_id):
        run = Run(f"{cls.__name__}/{run_id}")
        model = run.data.model
        return model

    @classmethod
    def predict(cls, dataset, run_id, model=None):
        run = Run(f"{cls.__name__}/{run_id}")
        run_config = run.data.config

        if model is None:
            model = self.load_trained_model(run_id)

        samples = []
        for sample in dataset:
            if 'label' not in sample:
                sample['label'] = None
            samples.append(Sample(**sample))

        label_lookup = run.data.label_lookup
        tokenizer = run.data.tokenizer
        encodings = OrderedDataset(
                run.data.tokenizer,
                samples,
                run_config['encode'],
                run.data.label_lookup
                )

        inverse_label_lookup = {v: k for k, v in run.data.label_lookup.items()}

        logger.info('Loading fine-tuned model')
        trainer = Trainer(
                model=run.data.model,
                data_collator=BatchCollator(run.data.tokenizer)
                )
        logger.info(f'Predicting labels for {len(samples)} samples')
        preds = trainer.predict(encodings)
        pred_probs = softmax(preds.predictions, axis=1)
        pred_labels = np.argmax(preds.predictions, axis=1)
        pred_labels = [inverse_label_lookup[p] for p in pred_labels]

        return pred_labels, pred_probs

    @step
    def start(self):
        tokenizer_config = self.config["tokenizer"]

        with open(self.documents_path, 'r') as f:
            self.documents = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)

        self.next(self.generate_label_lookup)

    @step
    def generate_label_lookup(self):
        """Maps sample labels to unique integer IDs and creates a lookup
        """
        self.label_lookup = dict()
        i = 0
        for doc in self.documents:
            label = doc['label']
            if label not in self.label_lookup:
                self.label_lookup[label] = i
                i += 1

        self.next(self.train_val_test_split)

    @step
    def train_val_test_split(self):
        """Splits the data into train, evaluation and test sets"""
        split_config = self.config["train_val_test_split"]
        train_size = split_config.pop('train_size')
        val_size = split_config.pop('val_size')
        test_size = split_config.pop('test_size')

        if sum([train_size, test_size, val_size]) > 1:
            raise ValueError('Sum of train, validation and test sizes must be \xe2 1')

        self.train_set, rest = train_test_split(self.documents, train_size=train_size,
                test_size=val_size + test_size, **split_config)
        self.val_set, self.test_set = train_test_split(rest, train_size=val_size, 
                test_size=test_size, **split_config)

        self.next(self.encode_train_set, self.encode_val_set)

    @step
    def encode_train_set(self):
        """Encodes the training dataset"""
        self.encodings = self._encode(self.train_set)
        self.next(self.encodings_join)

    @step
    def encode_val_set(self):
        """Encodes the evaluation dataset"""
        self.encodings = self._encode(self.val_set)
        self.next(self.encodings_join)

    @step
    def encodings_join(self, inputs):
        self.merge_artifacts(inputs, exclude=['encodings'])
        self.train_encodings = inputs.encode_train_set.encodings
        self.val_encodings = inputs.encode_val_set.encodings
        self.next(self.fine_tune)

    @step
    def fine_tune(self):

        model_config = self.config["model"]
        model_config["num_labels"] = len(self.label_lookup)
        logger.info('Loading pre-trained model')
        model = partial(model_init, model_config)
        training_args_config = self.config["training_args"]

        training_args = TrainingArguments(**training_args_config)
        trainer = Trainer(
                model=model(self.freeze_model),
                args=training_args,
                train_dataset=self.train_encodings,
                eval_dataset=self.val_encodings,
                compute_metrics=compute_metrics,
                data_collator=BatchCollator(self.tokenizer)
                )
        logger.info(f'Training model with {len(self.train_encodings)} samples')
        trainer.train()
        self.model = trainer.model

        trainer.save_model(training_args_config['output_dir'])
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    TrainTextClassifier()
