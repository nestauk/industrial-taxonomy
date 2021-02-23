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

import altair as alt

from industrial_taxonomy.getters.glass import get_organisation_description
from industrial_taxonomy.getters.companies_house import get_sector
from industrial_taxonomy.getters.glass_house import get_glass_house

# %%
transformer = 'stsb-distilbert-base'

# %%
from sklearn.decomposition import TruncatedSVD
from umap import UMAP

# %%
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(transformer)

# %% [markdown]
# ### 2.1 SIC Codes

# %%
from industrial_taxonomy.sic import (make_sic_lookups, save_sic_taxonomy, 
                                     load_sic_taxonomy, section_code_lookup, 
                                     extract_sic_code_description)

# %%
save_sic_taxonomy()
sic_2007 = load_sic_taxonomy()

# %%
sic4_lookup = extract_sic_code_description(sic_2007, 'Class')
sic4_structure_df = pd.read_csv('../../data/aux/sic4_2007_structure.csv')

# %%
sic4_structure_df[sic4_structure_df['description'].str.contains('spirits')]


# %%
def sic_description_preprocessing(descriptions):
    processed = []
    for description in descriptions:
        description = (description
                       .replace('/', ' ')
                       .replace('n.e.c.', '')
                       .replace('(', '')
                       .replace(')', '')
                       .replace('other', '')
                       .replace('-', '')
                      )
        
        tokens = description.split(' ')
        for exc in ['except', 'excluding']:
            if exc in tokens:
                except_index = tokens.index(exc)
                tokens = tokens[:except_index]
                description = ' '.join(tokens)
            
        processed.append(description)
        
    return processed


# %%
def get_product(sic4_structure):
    
    def takeout(description, activity):
        activities = activity.split(';')
        for activity in activities:
            description = description.lower()
            description = description.replace(activity.lower() + ' of', '')
            description = description.replace(activity.lower(), '')
        return description
    
    sic4_product = sic4_structure.apply(
        lambda row: takeout(row['description'], row['activity']),
        axis=1
    )
    
    sic4_product = sic_description_preprocessing(sic4_product)
    
    sic4_structure['product'] = sic4_product
    sic4_structure['product'] = sic4_structure['product'].str.lstrip().str.rstrip()
    
    return sic4_structure


# %%
sic4_structure_df = get_product(sic4_structure_df)

# %%
sic4_descriptions = sic_description_preprocessing(sic4_lookup.values())

# %%
sentence_embeddings = np.array(model.encode(sic4_structure_df['product']))

# %%
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

# %%
agg = AgglomerativeClustering(n_clusters=None, distance_threshold=25)
agg.fit(sentence_embeddings)

# %%
groups = defaultdict(list)

for description, label in zip(sic4_structure_df['product'], agg.labels_):
    groups[label].append(description)
    
groups_sorted = {k: groups[k] for k in 
                 sorted(groups, key=lambda k: len(groups[k]), reverse=True)}

# %%
product_group_lookup = {}

for k, products in groups_sorted.items():
    for product in products:
        product_group_lookup[product] = k

# %%
sic4_structure_df['product_group'] = (sic4_structure_df['product']
                                      .map(product_group_lookup))

# %%
sic4_structure_df.to_csv('../../data/aux/sic4_2007_structure.csv',
                         index=False)

# %%
