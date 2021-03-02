# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## 1. Preamble

# %%
# %run ../notebook_preamble.ipy

# %%
import yaml
import os

from sklearn.metrics import (classification_report, f1_score, 
                             confusion_matrix)
from imblearn.metrics import classification_report_imbalanced
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from industrial_taxonomy.getters.glass import get_organisation_description
from industrial_taxonomy.getters.companies_house import get_sector
from industrial_taxonomy.getters.glass_house import get_glass_house

# %%
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
with open(os.path.join(project_dir, 'industrial_taxonomy', 'model_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
    
# match_threshold = config['params']['match_threshold']
match_threshold = 75

# %% [markdown]
# ### 1.1 Load Data

# %%
glass_house = get_glass_house()
org_descriptions = get_organisation_description()
org_sectors = get_sector()

# %% [markdown]
# ### 1.2 Filter and Merge Data

# %%
org_descriptions

# %%
glass_house = glass_house[glass_house['score'] >= match_threshold]

orgs = org_sectors.merge(
    glass_house, left_on='company_number', 
    right_on='company_number', how='inner')
orgs = orgs.merge(
    org_descriptions, left_on='org_id', 
    right_on='org_id', how='inner')

# %%
orgs = (orgs
        .sort_values('score')
        .drop_duplicates('company_number', keep='last')
        .drop_duplicates('description', keep='last')
       )

# %%
orgs.info()

# %% [markdown]
# ## 2. EDA

# %% [markdown]
# ### 2.1 SIC Codes

# %%
from industrial_taxonomy.sic import load_sic_taxonomy, section_code_lookup, make_sic_lookups

# %%
for level in range(2, 5):
    orgs[f'SIC{level}_code'] = orgs['SIC5_code'].str[:level]

# %%
sic4_name_lookup, section_name_lookup, sic4_to_div_lookup = make_sic_lookups()
sic_2007 = load_sic_taxonomy()

orgs['SIC_section'] = orgs['SIC2_code'].map(section_code_lookup())

# %%
fig, ax = plt.subplots()
orgs['SIC_section'].value_counts().plot.bar(ax=ax)
ax.set_xlabel('SIC Section Code')
ax.set_ylabel('Frequency');

# %% [markdown]
# ### 2.2 Text Quality

# %%
fig, ax = plt.subplots()
ax.hist(orgs['description'].str.len(), bins=50)
ax.set_xlabel('N Characters')
ax.set_ylabel('Frequency');

# %% [markdown]
# ## 3 Preprocessing

# %% [markdown]
# ### 3.1 Train, Validate and Test Splitting

# %%
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    FRAC = .5
else:
    device = torch.device("cpu")
    FRAC = 0.001

# %%
orgs_sample = orgs.sample(frac=FRAC, random_state=RANDOM_SEED)

# %%
le = LabelEncoder()
sic_section_label = le.fit_transform(orgs_sample['SIC4_code'])

# label_section_lookup = {le.transform([k])[0]: v for k, v in section_name_lookup.items()}

# %%
N_LABELS = le.classes_.shape[0]

# %%
N_LABELS

# %%
TRAIN_SIZE = 0.8
VAL_SIZE=0.5

# %%
orgs_sample.shape

# %%
X_train, X_val, y_train, y_val = train_test_split(
    orgs_sample['description'],
    sic_section_label,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val,
    y_val,
    train_size=VAL_SIZE,
    random_state=RANDOM_SEED
)

X_train.shape, X_val.shape, X_test.shape

# %% [markdown]
# ## Dynamic Padding and Uniform Batching

# %%
from dataclasses import dataclass, field

@dataclass
class Row:
    text: str
    label: int


# %%
# from transformers.data.data_collator import DataCollator

# %%
import random
import time

from typing import Dict, Optional, List

import torch
from torch.utils.data.dataset import Dataset, IterableDataset
# from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoTokenizer, EvalPrediction, Trainer, 
                          HfArgumentParser, TrainingArguments, 
                          AutoModelForSequenceClassification, 
                          set_seed, AutoConfig)
from transformers import PreTrainedTokenizer, DataCollator, PreTrainedModel, PreTrainedTokenizerBase

@dataclass
class Features:
    input_ids: List[int]
    attention_mask: List[int]
    label: int
    
# class SICDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = [int(l) for l in labels]

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)
    
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
    
#     def __getitem__(self, i) -> Features:
#         return self.encode(row=self.rows[i])
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
    
def load_transformers_model(model: str, use_cuda: bool) -> PreTrainedModel:
    return model


# %%
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


# %%
MODEL = 'distilbert-base-uncased'
MAX_LEN = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = model_init()


# %%
def create_rows(text, labels):
    lens = [len(t) for t in text]
    _, text, labels = (list(t) for t in zip(*sorted(zip(lens, text, labels))))
    return [Row(t, l) for t, l in zip(text, labels)]


# %%
# args = TrainingArguments(output_dir=f"{project_dir}/models/test_dyn_padding",
#                          seed=RANDOM_SEED,
#                          num_train_epochs=1,
#                          per_device_train_batch_size=8,  # max batch size without OOM exception, because of the large max token length
#                          per_device_eval_batch_size=8,
# #                          evaluate_during_training=True,
#                          logging_steps=5000,
#                          save_steps=0,
#                         )

training_args = TrainingArguments(
    learning_rate=1e-5,
    output_dir=f'{project_dir}/models/sic4_bert_0',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=50,
    fp16=True,
    save_total_limit=2
#     do_train=True,
)

# %%
train = create_rows(X_train, y_train)
val = create_rows(X_val, y_val)

# %%
train_set = DynamicDataset(
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    rows=train,
    pad_to_max_length=False
)

# %%
val_set = DynamicDataset(
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    rows=val,
    pad_to_max_length=False
)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
#     data_collator=SmartCollator(pad_token_id=tokenizer.pad_token_id),
    data_collator=SmartCollator(tokenizer=tokenizer),
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
)

# %%
start_time = time.time()
loss = trainer.train()
print(f"training took {(time.time() - start_time) / 60:.2f}mn")
# result = trainer.evaluate()
# print(result)

# %%
trainer.save_model(f'{project_dir}/models/sic4_bert_0/model')

# %%
loss.metrics['train_runtime'] / 60

# %% [markdown]
# ### Load and Test

# %%
model = AutoModelForSequenceClassification.from_pretrained(
        f'{project_dir}/models/sic4_bert_0/model',
        num_labels=N_LABELS,
        return_dict=True)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
#     data_collator=SmartCollator(pad_token_id=tokenizer.pad_token_id),
    data_collator=SmartCollator(tokenizer=tokenizer),
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
)

# %%
test = create_rows(X_test, y_test)

# %%
test_set = DynamicDataset(
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    rows=test,
    pad_to_max_length=False
)

# %%
preds = trainer.predict(test_set)

# %%
pred_labels = np.argmax(preds.predictions, axis=1)
pred_codes = le.inverse_transform(pred_labels)


# %%
def print_description_preds(dataset, pred_codes, topn=10, first=17000):
    descriptions = [row.text for row in dataset.rows[first:first+topn]]
    labels = pred_codes[first:first+topn]
    
    for d, l in zip(descriptions, labels):
        print('- Pred:', sic4_name_lookup[l])
        print('- Text:', d[:200] + '...', '\n')


# %%
print_description_preds(test_set, pred_codes, topn=20)


# %%
def create_pred_table(rows, preds, label_encoder):
    records = []
    preds = label_encoder.inverse_transform(preds)
    labels = label_encoder.inverse_transform([r.label for r in rows])
    text = [r.text for r in rows]
    
    return pd.DataFrame({'text': text, 'label': labels, 'pred': preds})


# %%
pred_table = create_pred_table(test, pred_labels, le)

# %%
pred_table.to_csv('../../data/interim/sic4_test_predictions.csv', index=False)

# %%
from sentence_transformers import SentenceTransformer
bert_model = 'stsb-distilbert-base'
st = SentenceTransformer(bert_model)

# %%
test_encodings = st.encode(pred_table['text'])

# %%
np.save('../../data/interim/sic4_test_embeddings.np', test_encodings)

# %% [markdown]
# ### 3.2 Tokenization

# %%
from transformers import DistilBertTokenizerFast, AutoTokenizer

MODEL = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# %%
tokenization_args = {
    'max_length': 256,
    'truncation': True,
    'padding': True,
}

train_encodings = tokenizer(list(X_train), **tokenization_args)
val_encodings = tokenizer(list(X_val), **tokenization_args)
test_encodings = tokenizer(list(X_test), **tokenization_args)


# %%
class SICDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = [int(l) for l in labels]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    
train_dataset = SICDataset(train_encodings, y_train)
val_dataset = SICDataset(val_encodings, y_val)
test_dataset = SICDataset(test_encodings, y_test)

# %% [markdown]
# ## 4 HuggingFace

# %% [markdown]
# ### 4.1 Model Creation

# %%
from transformers import (Trainer, TrainingArguments, 
                          AutoModelForSequenceClassification,
                          AutoConfig,
                         )

from torch import nn
from sklearn.metrics import (accuracy_score, 
                             precision_recall_fscore_support)


# %%
def model_init(frozen=False):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=N_LABELS,
        return_dict=True)
#     if frozen:
#         for param in model.base_model.parameters():
#             param.requires_grad = False
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


# %%
model = model_init()

# training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
#     logging_steps=20,
#     eval_steps=5,
#     save_total_limit=2,
#     save_steps=5000,
#     seed=RANDOM_SEED,
#     per_device_eval_batch_size=8,
#     per_device_train_batch_size=4,
#     logging_first_step=True,
#     learning_rate=5e-5,
#     num_train_epochs=2,
# )

training_args = TrainingArguments(
    learning_rate=1e-5,
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
#     do_train=True,
)

trainer = Trainer(
    args=training_args,
#     tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
#     model_init=model_init,
#     model=model_init(),
    model=model,
    compute_metrics=compute_metrics
)

# %%
training_loss = trainer.train()

# %%
trainer.save_model(f'{project_dir}/models/sic_section_0/model')

# %%
fig, ax = plt.subplots()
ax.plot(training_loss)
ax.set_xlabel('Training Batch')
ax.set_ylabel('Training Loss')
plt.savefig(f'{project_dir}/models/sic_section_0/figures/training_loss_vs_batch_sic_section_bert-base-uncased_line.png', dpi=300);

# %%
loss = trainer.evaluate()

# %%
loss

# %%
preds = trainer.predict(test_dataset)
pred_sectors = le.inverse_transform(np.argmax(preds.predictions, axis=1))
pred_labels = np.argmax(preds.predictions, axis=1)


# %%
def save_classification_report(test, preds, path, imbalanced=False, dp=2, label_lookup=None, fmt='csv'):
    
    def lookup_label(label):
        replace = label_lookup.get(label)
        if replace is None:
            if label.isdigit():
                replace = label_lookup.get(int(label))
                return replace
            else:
                return label
        
    if imbalanced:
        report_fn = classification_report_imbalanced
    else:
        report_fn = classification_report
    results = report_fn(y_test, preds, output_dict=True)
    results_df = pd.DataFrame.from_records(results).T
    for col in results_df.columns:
        if not col == 'support':
            results_df[col] = results_df[col].round(dp)
    results_df = (results_df
                  .reset_index()
                  .rename(columns={'index': 'label'}))
    
    if label_lookup is not None:
        results_df['label'] = results_df['label'].apply(lookup_label)
    
    if fmt == 'csv':
        results_df.to_markdown(path, index=False)
    elif fmt == 'md':
        results_df.to_markdown(path, index=False)
        
    return results_df


# %%
_ = save_classification_report(y_test, pred_labels,
                           path=f'{project_dir}/models/sic_section_0/classification_report.md', 
                           fmt='md', label_lookup=label_section_lookup)

# %%
print(classification_report(y_test, pred_labels))

# %%
print(classification_report_imbalanced(y_test, pred_labels))

# %%
import seaborn as sns

# %%
fig, ax = plt.subplots(figsize=(10, 10))
cm = confusion_matrix(y_test, pred_labels)
cm = pd.DataFrame(cm, columns=label_section_lookup.values(), index=label_section_lookup.values())
cm = cm.divide(cm.sum(axis=1), axis=0) * 100
sns.heatmap(cm, ax=ax, annot=True, fmt='.1f', square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel('Predicted SIC Section')
ax.set_ylabel('True SIC Section')
plt.savefig(f'{project_dir}/models/sic_section_0/figures/confusion_matrix_norm.png', dpi=150);


# %%
def print_classification_examples(X, y, preds, pred_type='tp', num_examples=5, 
                                  random_state=None, label_lookup=None, max_chars=None):
    
    labels, label_counts = np.unique(y, return_counts=True)
    pred_labels, pred_label_counts = np.unique(preds, return_counts=True)
    X_ = np.array(X)
    
    for label in sorted(labels):
        if label_lookup is not None:
            print('=== Label:', label_lookup[label], '===\n')
        else:
            print('=== Label:', label, '===\n')
        if pred_type == 'tp':
            sample_ids = np.argwhere((preds == label) & (y == label))
        elif pred_type == 'fp':
            sample_ids = np.argwhere((preds == label) & (y != label))
        elif pred_type == 'tn':
            sample_ids = np.argwhere((preds != label) & (y != label))
        elif pred_type == 'fn':
            sample_ids = np.argwhere((preds != label) & (y == label))
            
        sample_ids = sample_ids.ravel()[:num_examples]
        for i in sample_ids:
            if label_lookup is not None:
                print('True:', label_lookup[y[i]])
                print('Pred:', label_lookup[preds[i]])
            else:
                print('True:', y[i])
                print('Pred:', preds[i])
                
            if max_chars is not None:
                print('Description:', X_[i][:max_chars] + '...', '\n')
            else:
                print('Description:', X_[i], '\n')
            
        print('')


# %%
print_classification_examples(X_test, y_test, pred_labels, pred_type='fp', 
                              label_lookup=label_section_lookup, max_chars=200)

# %% [markdown]
# ## Hyperparameter Tuning

# %%
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from typing import List

import numpy as np
import torch
from torch.utils.data.dataset import Dataset, IterableDataset
# from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoTokenizer, EvalPrediction, Trainer, HfArgumentParser, TrainingArguments,
    AutoModelForSequenceClassification, set_seed, AutoConfig,)
from transformers import PreTrainedTokenizer, DataCollator, PreTrainedModel
# import wandb 

# %%
training_args = dict(output_dir = './models',
overwrite_output_dir = True,
save_steps = 0,
seed = 321,
num_train_epochs = 1,
learning_rate = 5e-5,
per_gpu_train_batch_size = 64,
gradient_accumulation_steps = 1,
per_gpu_eval_batch_size = 64,
fp16 = True,
# evaluate_during_training = True
)

model_params = dict(max_seq_len = 256,
dynamic_padding = True,
smart_batching = True)

# %%
training_args = TrainingArguments(**training_args)
model_args = ModelParameters(**model_params)

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# %%
if model_args.max_seq_len:
    max_sequence_len = model_args.max_seq_len
else:
    longest_sentence = max(train_sentences, key=len)
    max_sequence_len = len(tokenizer.encode(text=longest_sentence.text_a))

# %%
# parser = HfArgumentParser((TrainingArguments, ModelParameters))
# training_args, model_args = parser.parse_args_into_dataclasses()  # type: (TrainingArguments, ModelParameters)

# train_sentences = load_train_data(path="resources/XNLI-MT-1.0/multinli/multinli.train.fr.tsv",
#                                   sort=model_args.smart_batching)

train_batches = build_batches(sentences=train_sentences, batch_size=training_args.per_gpu_train_batch_size)
valid_sentences = load_dev_data(path="resources/XNLI-1.0/xnli.test.tsv")
valid_batches = build_batches(sentences=valid_sentences, batch_size=training_args.per_gpu_eval_batch_size)

train_set = DynamicSICDataset(tokenizer=tokenizer,
                        max_len=max_sequence_len,
                        examples=train_batches,
                        pad_to_max_length=not model_args.dynamic_padding)

valid_set = DynamicSICDataset(tokenizer=tokenizer,
                        max_len=max_sequence_len,
                        examples=valid_batches,
                        pad_to_max_length=not model_args.dynamic_padding)

model = load_transformers_model(pretrained_model_name_or_path="camembert-base",
                                use_cuda=True,
                                mixed_precision=False)


def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    # data_collator=IdentityCollator(pad_token_id=tokenizer.pad_token_id),
    data_collator=SmartCollator(pad_token_id=tokenizer.pad_token_id),
#         tb_writer=SummaryWriter(log_dir='logs', flush_secs=10),
    eval_dataset=valid_set,
    compute_metrics=compute_metrics,
)

# %%
start_time = time.time()
trainer.train()
# wandb.config.update(model_args)
# wandb.config.update(training_args)
# wandb.log({"training time": int((time.time() - start_time) / 60)})
trainer.save_model()
trainer.evaluate()
logging.info("*** Evaluate ***")
result = trainer.evaluate()
# wandb.log(result)

output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    logging.info("***** Eval results *****")
    for key, value in result.items():
        logging.info("  %s = %s", key, value)
        writer.write("%s = %s\n" % (key, value))


# %%
print(classification_report(y_test, pred_labels))

# %%
for i, c  in zip(X_test[y_test == 0][:10].index, X_test[y_test == 0][:10].values):
    print(i, c, '\n')

# %%
for i_batch, (X, y) in enumerate(test_dataset):
#     X = X.to(device)
#     y = y.to(device)

    y_pred = trainer.predict(X) # in eval model we get the softmax output so, don't need to index


    y_pred = torch.argmax(y_pred, dim = -1)

    print(y)
    print(y_pred)
    print('--------------------')

# %%
import ray
from ray import tune


# %%
def hp_space(trial):

    return {
        "learning_rate": tune.loguniform(1e-6, 5e-5),
        "num_train_epochs": tune.choice(range(1, 3)),
        "per_device_train_batch_size": tune.choice([4, 8, 16]),
        "per_device_eval_batch_size": tune.choice([8, 16, 32]),
        }


# %%
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

# %%
ray.init()

# %%
best_run = trainer.hyperparameter_search(
    direction='maximize',
    backend='ray',
#     search_alg=HyperOptSearch(),
    # Choose among schedulers:
    # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
    scheduler=AsyncHyperBandScheduler(
        metric='eval_f1',
        mode='max',
        grace_period=1, 
        max_t=2,
    ),
    n_trials=2,
#     n_jobs=1,
    hp_space=hp_space,
);

# %%
ray.shutdown()

# %%
