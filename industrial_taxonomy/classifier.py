from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments)
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
from torch.utils.data.dataset import Dataset

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


@dataclass
class Features:
    input_ids: List[int]
    attention_mask: List[int]
    label: int
    
class DynamicDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                pad_to_max_length: bool, max_len: int,
                rows: List[Row]) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rows: List[Rows] = rows
        self.current = 0
        self.pad_to_max_length = pad_to_max_length
        
    def encode(self, row: Row) -> Features:
        encode_dict = self.tokenizer(
            text=row.text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=self.pad_to_max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False
        )
        return Features(input_ids=encode_dict['input_ids'],
                        attention_mask=encode_dict['attention_mask'],
                        label=row.label
                       )
    
    def __getitem__(self, _) -> Features:
        if self.current == len(self.rows):
            self.current = 0
        row = self.rows[self.current]
        self.current += 1
        return self.encode(row)

    def __len__(self):
        return len(self.rows)
        
def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]

@dataclass
class SmartCollator():
    tokenizer: PreTrainedTokenizerBase
    
    def __call__(self, batch):
        return self._collate_batch(batch, self.tokenizer)
        
    def _collate_batch(self, batch: List[Features], tokenizer) -> Dict[str, torch.Tensor]:
        batch_inputs = list()
        batch_attn_masks = list()
        labels = list()
        max_size = max([len(item.input_ids) for item in batch])
        for item in batch:
            batch_inputs += [pad_seq(
                item.input_ids, max_size, tokenizer.pad_token_id)]
            batch_attn_masks += [pad_seq(
                item.attention_mask, max_size, 0)]
            labels.append(item.label)
            
        return {
            'input_ids': torch.tensor(batch_inputs, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attn_masks, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
               }
   
def model_init(frozen=False):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=N_LABELS,
        return_dict=True)
    if frozen:
        for param in model.base_model.parameters():
            param.requires_grad = False
    return model

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='micro'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }   

def load_trained_model(path, tokenizer, training_args)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, return_dict=True)

    trainer = Trainer(
	model=model,
	args=training_args,
	data_collator=SmartCollator(tokenizer=tokenizer),
	compute_metrics=compute_metrics,
    )
    return trainer
