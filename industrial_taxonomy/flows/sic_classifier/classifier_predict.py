"""Trains a SIC code classifier"""


from metaflow import FlowSpec, step, Parameter, JSONType


class TuneClassifier(FlowSpec):
    num_labels = Parameter(
            "number of labels",
            help="Number of classes in classification",
            type=int,
            default=2
            )
    random_seed = Parameter(
            "random seed",
            help="A random seed to split the data.",
            type=int,
            default=None
            )
    train_size = Parameter(
            "train size",
            help="Fraction of samples to use for training.",
            type=float,
            default=.8
            )
    trainer_params = Parameter(
            "trainer params",
            help="Config for transformers.Trainer",
            type=JSONType,
            )
    training_args = Parameter(
            "training args",
            help="Config for transformers.TrainingArguments",
            type=JSONType,
            )

