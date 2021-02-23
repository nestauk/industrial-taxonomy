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
with open(os.path.join(project_dir, 'industrial_taxonomy', 'model_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
    
match_threshold = config['params']['match_threshold']

# %% [markdown]
# ### 1.1 Load Data

# %%
glass_house = get_glass_house()
org_descriptions = get_organisation_description()
org_sectors = get_sector()

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
for level in range(1, 5):
    orgs[f'SIC{level}_code'] = orgs['SIC5_code'].str[:level]

# %%
from industrial_taxonomy.sic import make_sic_lookups, save_sic_taxonomy, load_sic_taxonomy, section_code_lookup, extract_sic_code_description
from operator import itemgetter

# %%
sic_2007 = load_sic_taxonomy()

# %%
orgs['SIC_section'] = orgs['SIC2_code'].map(section_code_lookup())

# %%
fig, ax = plt.subplots()
orgs['SIC_section'].value_counts().plot.bar(ax=ax)
ax.set_xlabel('SIC Section Code')
ax.set_ylabel('Frequency');

# %%
orgs['SIC_section'].value_counts()

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
# ## 3 Modelling

# %%
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 6, 4

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0")

PRE_TRAINED_MODEL_NAME = 'distilbert-base-cased'

# %%
codes = ['8299', '7022', '9609', '9999']

# %% [markdown]
# ### With HuggingFace

# %% [markdown]
# ### 3.2 Train, Validate and Test Splitting

# %%
orgs = orgs.sample(frac=.01, random_state=RANDOM_SEED)
# orgs['SIC4_code'] = orgs['SIC4_code'].astype(np.int32)

n_train = np.int32(np.round(0.9 * orgs.shape[0]))
orgs_test = orgs.iloc[n_train:]

# %%
orgs.shape

# %%
orgs_train, orgs_val = train_test_split(
  orgs.iloc[:n_train],
  test_size=0.2,
  random_state=RANDOM_SEED
)

orgs_train.shape, orgs_val.shape, orgs_test.shape

# %%
from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased')

# %%
train_encodings = tokenizer(list(orgs_train['description']), max_length=256, 
                            truncation=True, padding=True)
val_encodings = tokenizer(list(orgs_val['description']), max_length=256, 
                          truncation=True, padding=True)
test_encodings = tokenizer(list(orgs_test['description']), max_length=256, 
                           truncation=True, padding=True,)

# %%
code_label_map = {c: i for i, c in enumerate(orgs['SIC_section'].unique())}    

# %%
import torch

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

train_dataset = SICDataset(
    train_encodings, 
    orgs_train['SIC_section'].map(code_label_map))
val_dataset = SICDataset(
    val_encodings,
    orgs_val['SIC_section'].map(code_label_map))
test_dataset = SICDataset(
    test_encodings, 
    orgs_test['SIC_section'].map(code_label_map))

# %%
# TrainingArguments??

# %%
from transformers import (DistilBertForSequenceClassification, 
                          Trainer, TrainingArguments)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=4,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    save_total_limit=1,
    learning_rate=1e-4,
    
)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=21
)


# %%
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
)

# %%
trainer.train()

# %%
from sklearn.metrics import classification_report

# %%
preds = trainer.predict(test_dataset)
y_test = [int(test_dataset[i]['labels']) for i in range(len(test_dataset))]

# %%
print(classification_report(y_test, np.argmax(preds.predictions, axis=1)))

# %%
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()

# %% [markdown]
# ### 3.1 Tutorial

# %%
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# %%
token_lens = []

for txt in orgs['description'].sample(1000):
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

# %%
fig, ax = plt.subplots()
ax.hist(token_lens, bins=50)
ax.set_xlabel('N Tokens')
ax.set_ylabel('Frequency');

# %%
MAX_LEN = 512


# %%
class OrgDataset(Dataset):
    def __init__(self, descriptions, targets, tokenizer, max_len):
        self.descriptions = descriptions
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        description = str(self.descriptions[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
          description,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
            truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        return {
          'description_text': description,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }


# %%
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = OrgDataset(
        descriptions=df['description'].to_numpy(),
        targets=df['SIC4_code'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
  )

BATCH_SIZE = 8

train_data_loader = create_data_loader(
    orgs_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(
    orgs_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(
    orgs_test, tokenizer, MAX_LEN, BATCH_SIZE)

# %%
data = next(iter(train_data_loader))

data.keys()

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

# %%
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


# %%
class SICClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SICClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


# %%
model = SICClassifier(len(codes))
model = model.to(device)

# %%
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

# %% [markdown]
# ### 3.3 Training

# %%
EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


# %%
def train_epoch(model, data_loader, loss_fn, 
                optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


# %%
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
            input_ids=input_ids,

            attention_mask=attention_mask

            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


# %%
torch.cuda.empty_cache()

# %%
# %%time

history = defaultdict(list)

best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(orgs_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(orgs_val)

  )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

# %%
