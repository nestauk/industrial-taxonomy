"""Fine tune a transformer for classification using uniform batching and 
dynamic padding.
"""

import json
from dataclasses import dataclass, field

from metaflow import FlowSpec, step, Parameter, JSONType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from industrial_taxonomy.flows.sic_classifier.classifier_utils import (
       BatchCollator, OrderedDataset, compute_metrics, model_init) 
from industrial_taxonomy.sic import extract_sic_code_description, load_sic_taxonomy


class TrainTextClassifier(FlowSpec):
    documents_path = Parameter(
            "documents path",
            help="Path to JSON training data",
            type=str,
            )
    output_dir = Parameter(
            "output_dir",
            help="Path to directory for saving model",
            type=str
            )
    sic_level = Parameter(
            "sic_level",
            help="Level of SIC to use as training and prediction classes",
            type=str
            )
    freeze_model = Parameter(
            "freeze model",
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

    @step
    def start(self):
        model_config = self.config["model"]
        tokenizer_config = self.config["tokenizer"]

        with open(self.documents_path, 'r') as f:
            self.documents = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(**self.tokenizer_config)
        self.model = partial(model_init, self.freeze_model, self.model_config)

        self.next(self.generate_classes)


    @step
    def generate_class_lookup(self):

        sic_level = self.config["sic_level"]
        sic_2007 = load_sic_taxonomy()
        sic_lookup = extract_sic_code_description(sic_2007, sic_level)

        self.class_lookup = {c: i for i, c in enumerate(sic_lookup.keys())}
        self.next(self.train_validation_split)


    @step
    def train_val_test_split(self):
        split_config = self.config["train_val_test_split"]
        train_size = split_config.pop('train_size')
        val_size = split_config.pop('val_size')
        test_size = split_config.pop('test_size')

        if sum([train_size, test_size, val_size]) > 1:
            raise ValueError('Sum of train, validation and test sizes must be ≤ 1')

        self.train_set, rest = train_test_split(self.documents, train_size=train_size,
                test_size=val_size + test_size, **split_config)
        self.val_set, self.test_set = train_test_split(rest, train_size=val_size, 
                test_size=test_size, **split_config)

        self.next(self.prepare_train_dataset, self.prepare_val_dataset)

    @step
    def encode_train_set(self):
        encode_config = self.config["encode"]
        self.encodings = OrderedDataset(self.tokenizer, 
                self.train_set, encode_config, self.class_lookup)
        self.next(self.encodings_join)

    @step
    def encode_val_set(self):
        encode_config = self.config["encode"]
        self.encodings = OrderedDataset(self.tokenizer, 
                self.val_set, encode_config)
        self.next(self.encodings_join, self.class_lookup)

    @step
    def encodings_join(self):
        self.train_encodings = self.encode_train_set.encodings
        self.val_encodings = self.encode_val_set.encodings
        self.next(fine_tune)

    @step
    def fine_tune(self):
        training_args_config = self.config["training_args"]
        trainer_config = self.config["trainer"]

        training_args = TrainingArguments(**training_args_config)
        trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_encodings,
                eval_dataset=self.val_encodings,
                compute_metrics=compute_metrics,
                data_collator=BatchCollator(self.tokenizer)
                )
        trainer.train()

        trainer.save_model(self.output_dir)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    TrainTextClassifier()
