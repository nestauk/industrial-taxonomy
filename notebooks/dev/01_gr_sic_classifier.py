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
#       jupytext_version: 1.9.1
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

from industrial_taxonomy.getters.glass import get_organisation_description
from industrial_taxonomy.getters.companies_house import get_sector
from industrial_taxonomy.getters.glass_house import get_glass_house

# %%
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
with open(os.path.join(project_dir, 'industrial_taxonomy', 'model_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
    
match_threshold = config['params']['match_threshold']

# %% [markdown]
# ### 1.1 Load Data

# %%
glass_house = get_glass_house()
org_descriptions = get_organisation_description()
org_sectors = get_sector()

# %%
# del glass_house
# del org_sectors
# del org_descriptions

# %% [markdown]
# ### 1.2 Filter and Merge Data

# %%
glass_house = glass_house[glass_house['score'] >= match_threshold]
org_descriptions = org_descriptions.drop_duplicates(subset='org_id', keep='last')

orgs = org_sectors.merge(
    glass_house, left_on='company_number', 
    right_on='company_number', how='inner')
orgs = orgs.merge(
    org_descriptions, left_on='org_id', 
    right_on='org_id', how='inner')

# %% [markdown]
# ## 2. EDA

# %%
orgs.info()

# %% [markdown]
# ### 2.1 SIC Codes

# %%
for level in range(2, 5):
    orgs[f'SIC{level}_code'] = orgs['SIC5_code'].str[:level]

# %%
from industrial_taxonomy.sic import load_sic_taxonomy, section_code_lookup

# %%
sic_2007 = load_sic_taxonomy()

# %%
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

# %%
quantiles = np.arange(0, 1.01, 0.01)

fig, ax = plt.subplots()
ax.plot(quantiles, np.quantile(orgs['description'].str.len(), quantiles))
ax.set_xlabel('Quantile')
ax.set_ylabel('N Characters')
ax.set_yscale('log');

# %% [markdown]
# ## 3 Preprocessing

# %% [markdown]
# ### 3.1 Train, Validate and Test Splitting

# %%
orgs_sample = orgs.sample(frac=.001, random_state=RANDOM_SEED)

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %%
le = LabelEncoder()
sic_section_label = le.fit_transform(orgs_sample['SIC_section'])

# %%
N_LABELS = le.classes_.shape[0]

# %%
TRAIN_SIZE = 0.5
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
# ### 3.2 Tokenization

# %%
from transformers import DistilBertTokenizerFast

MODEL = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL)

# %%
# from transformers import DistilBertTokenizerFasttokenizer = DistilBertTokenizer.from_pretrained(
#     'distilbert-base-uncased')

# %%
tokenization_args = {
    'max_length': 265,
    'truncation': True,
    'padding': True,
}

train_encodings = tokenizer(list(X_train), **tokenization_args)
val_encodings = tokenizer(list(X_val), **tokenization_args)
test_encodings = tokenizer(list(X_test), **tokenization_args)

# %%
# import tensorflow as tf
import torch

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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

# %%
# train_dataset = tf.data.Dataset.from_tensor_slices(
#     (dict(train_encodings), y_train)
# )
# val_dataset = tf.data.Dataset.from_tensor_slices(
#     (dict(val_encodings), y_val)
# )
# test_dataset = tf.data.Dataset.from_tensor_slices(
#     (dict(test_encodings), y_test)
# )

# %% [markdown]
# ## 4 HuggingFace + Ray

# %% [markdown]
# ### 4.1 Model Creation

# %%
from transformers import (Trainer, TrainingArguments, 
                          AutoModelForSequenceClassification,
                          AutoConfig,
#                           BertForSequenceClassification, 
#                           DistilBertConfig
                         )

from torch import nn
from sklearn.metrics import (accuracy_score, 
                             precision_recall_fscore_support)

# %%
model = AutoModelForSequenceClassification.from_pretrained(
        MODEL)


# %%
# for name, param in model.named_parameters():
#     if 'classifier' not in name: # classifier layer
#         param.requires_grad = False

# %%
def model_init(frozen=True):
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
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=500,
#     eval_steps=5,
    save_total_limit=2,
    save_steps=5000,
    seed=RANDOM_SEED,
    evaluation_strategy='steps',
    per_device_eval_batch_size=8,
    per_device_train_batch_size=4,
    logging_first_step=True,
    do_eval=True,
    do_train=True,
    
)

trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    model_init=model_init,
#     model=model_init(),
    compute_metrics=compute_metrics
)

# %%
loss = trainer.evaluate()

# %%
loss

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
