"""Tools for fine-tuning a transformer"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
from torch.utils.data.dataset import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# from industrial_taxonomy.getters.glass import get_organisation_description
# from industrial_taxonomy.getters.companies_house import get_sector
# from industrial_taxonomy.getters.glass_house import get_glass_house
from industrial_taxonomy.utils.metaflow_client import cache_getter_fn


@dataclass
class Sample:
    """Sample of raw data

    Attributes:
        index (int): A UID for the sample.
        text (str): Raw or preprocessed text.
        label (int): Integer label of class from labelled data.
    """

    index: int
    text: str
    label: int


@dataclass
class Features:
    """Dataclass structure.

    Holds sequence features and label.

    Attributes:
        input_ids (list): Token IDs from a sequence.
        attention_mask (list): Token attention mask for a sequence.
        label (int): Integer label of class from labelled data.
    """

    input_ids: List[int]
    attention_mask: List[int]
    label: int
    index: int


class OrderedDataset(Dataset):
    """A dataset that returns samples in the order that they were inserted."""

    def __init__(
        self,
        tokenizer: PreTrainedModel,
        samples: List[Sample],
        config: dict,
        label_lookup: dict,
    ) -> None:
        """
        Args:
            tokenizer (transformers.PreTrainedTokenizer): Pretrained tokenizer
                matching the transformer model in use.
            samples (list): The samples to be included in the dataset
        """
        self.tokenizer = tokenizer
        self.label_lookup = label_lookup
        self.config = config
        self.current = 0
        self.features = self._sort_samples(samples)

    def _sort_samples(self, samples):
        """Encodes and sorts samples"""
        features = [self._encode(s) for s in samples]
        lens = [len(f.input_ids) for f in features]
        features = [f for _, f in sorted(zip(lens, features), key=lambda t: t[0])]
        return features

    def _encode(self, sample: Sample) -> Features:
        """Tokenizes and encodes a sequence.

        Arguments:
            sample (Sample): A sample with a sequence to be encoded.

        Returns:
            Features: Encoded sequence.
        """
        encode_dict = self.tokenizer(text=sample.text, **self.config)

        if self.label_lookup is not None:
            label = self.label_lookup.get(sample.label, 0)
        else:
            label = sample.label

        return Features(
            input_ids=encode_dict["input_ids"],
            attention_mask=encode_dict["attention_mask"],
            label=label,
            index=sample.index,
        )

    def __getitem__(self, _) -> Features:
        """Gets the next encoded sample.

        This method always returns the next sample based on their original
        ordering. It overrides any input index that may be attempting to fetch
        a specific sample.

        Samples are returned encoded.
        """
        if self.current == len(self.features):
            self.current = 0
        sample = self.features[self.current]
        self.current += 1
        return sample

    def __len__(self):
        return len(self.features)


def pad_seq(seq: List[int], max_len: int, pad_value: int) -> List[int]:
    """Pads a tokenized sequence to the maximum length with a filler token.

    Tokenized sequences with len < max_len are padded up to max_len with a
    pre-selected token ID.
    """
    return seq + (max_len - len(seq)) * [pad_value]


@dataclass
class BatchCollator:
    """Collates a batch of samples for a Trainer.

    Attributes:
        tokenizer: The tokenizer class being used for sequence encoding.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, batch):
        return self._collate_batch(batch, self.tokenizer)

    def _collate_batch(
        self, batch: List[Features], tokenizer
    ) -> Dict[str, torch.Tensor]:
        batch_inputs = list()
        batch_attn_masks = list()
        labels = list()
        max_size = max([len(item.input_ids) for item in batch])
        for item in batch:
            batch_inputs += [pad_seq(item.input_ids, max_size, tokenizer.pad_token_id)]
            batch_attn_masks += [pad_seq(item.attention_mask, max_size, 0)]
            labels.append(item.label)

        return {
            "input_ids": torch.tensor(batch_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def model_init(model_kwargs, frozen=False):
    model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
    if frozen:
        for param in model.base_model.parameters():
            param.requires_grad = False
    return model


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def create_org_data(match_threshold, sic_level=4):
    glass_house = get_glass_house()
    org_descriptions = get_organisation_description()
    org_sectors = get_sector()

    glass_house = glass_house[glass_house["score"] >= match_threshold]

    orgs = org_sectors.merge(
        glass_house, left_on="company_number", right_on="company_number", how="inner"
    )
    orgs = orgs.merge(
        org_descriptions, left_on="org_id", right_on="org_id", how="inner"
    )

    orgs = (
        orgs.sort_values("score")
        .drop_duplicates("company_number", keep="last")
        .drop_duplicates("description", keep="last")
    )

    orgs[f"SIC_code"] = orgs[f"SIC5_code"].str[:sic_level]
    orgs = orgs[["description", f"SIC_code"]]
    orgs = orgs.reset_index().rename(
        columns={"description": "text", "SIC_code": "label"}
    )
    return orgs.to_dict(orient="records")
