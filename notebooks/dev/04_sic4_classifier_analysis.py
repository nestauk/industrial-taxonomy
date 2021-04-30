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

# %%
# %run ../notebook_preamble.ipy

# %%
import joblib

from umap import UMAP
from sklearn.decomposition import TruncatedSVD

from industrial_taxonomy.sic import make_sic_lookups

import seaborn as sns

# %%
fig_save_opts = dict(dpi=300, bbox_inches='tight')

# %%
sns.set_context('notebook')

# %%
sic4_name_lookup, section_name_lookup, sic4_to_div_lookup = make_sic_lookups()

# %% [markdown]
# ### Questions:
#
# - [ ] What codes does the model struggle to identify?
#     - [x] Codes which are underrepresented in relation to their presence in CH
#     - [x] What types of codes are these?
# - [ ] How noisy is the labelling in CH?
#     - [ ] What is the semantic distance between company descriptions and their CH labels?
# - [ ] How challenging was it for the model to classify certain companies or codes?
#     - What true codes have the widest distribution of predicted codes?
# - [ ] How coherent/homogeneous are the companies within a certain part of the taxonomy?
#     - Calculate the silhouette score for each group of companies based on their true SIC code.
# - [ ] What is the qualitative difference between the predicted and true SIC code where they differ?
#     - [ ] Are they in different parts of the taxonomy?
#     - [ ] Are they more specific/general? (when they are just 1 move away in the taxonomy)
# - [ ] What is the semantic relationship between codes and descriptions?
#     - [ ] How far are company descriptions that are wrongly labelled to the ones that are truly labelled?
#     - [ ] Between the codes of incorrectly labelled?
#     - [ ] Between the codes and descriptions of all labelled?
# - [ ] Limitations of the current taxonomy
#     - [ ] Why are these codes difficult to classify? Is it the company descriptions or the limitations of the codes?
# - [ ] How do we reverse engineer some of the sources of error to find out where they have happened?
#     - [ ] Train a text generation or augmentation algorithm to more succesfully predict the true label? How much does this language differ?
# - [ ] What are the limitations of the current codes for describing the companies?
#   - [ ] Codes that are predicted together but are semantically disimilar.
#     
#     
# Look at the average semantic similarity between the group that was predicted and the group that is the true label

# %%
le = joblib.load('../../models/sic4_bert_1/label_encoder.pkl')

# %%
test_df = pd.read_csv(f'{data_path}/processed/sic4_test_predictions.csv')
pred_probs = np.load(f'{data_path}/processed/sic4_test_pred_probs.npy')
test_encodings = np.load(f'{data_path}/processed/sic4_test_embeddings.npy')
sic4_encodings = np.load(f'{data_path}/processed/sic4_encodings.npy')
train_df = pd.read_csv(f'{data_path}/processed/train_data.csv')

test_df['label'] = test_df['label'].astype(str).str.zfill(4)
test_df['pred'] = test_df['pred'].astype(str).str.zfill(4)
train_df['label'] = train_df['label'].astype(str).str.zfill(4)

# %%
test_df.head()

# %%
top_n = 20

label_counts = test_df['label'].value_counts()
pred_counts = test_df['pred'].value_counts()
pred_ratio = pred_counts / label_counts
pred_ratio = pred_ratio.sort_values()

fig, axs = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)

pred_ratio.dropna()[-top_n:].plot.barh(ax=axs[0])
pred_ratio[:top_n].plot.barh(ax=axs[1], color='C3')

for ax in axs.ravel():
    ax.axvline(1, color='gray', linestyle='--')
    y_tick_labels = ax.get_yticklabels()
    for t in y_tick_labels:
        t.set_text(sic4_name_lookup.get(t.get_text()))        
    ax.set_yticklabels(y_tick_labels)
    
ax.set_xlabel('Ratio of Predicted Codes to Labelled Codes')

plt.tight_layout()

plt.savefig(f'{project_dir}/figures/predicted_sic4_representation_barh.png', **fig_save_opts)

# %% [markdown]
# We can see that there are certain labels which are highly over-represented in the predictions. These include several so-called 'n.e.c.' codes that pertain to industries that are not captured by other labels. This suggests that many business activities are not easily categorised by a specific code from their descriptions and that assigning them a more generalised label gives a lower loss. 

# %%
test_df['pred'].unique().shape

# %%
pred_ratio_sic4 = pd.Series(pred_ratio.index.map(sic4_name_lookup), index=pred_ratio.index)

# %%
general_labels = (pred_ratio_sic4.str.contains('n.e.c.')
#                   | pred_ratio_sic4.str.lower().str.contains('other')
                 )

# %%
general_labels.sum()

# %%
general_labels[pred_ratio > 1]

# %%
svd_sic4 = TruncatedSVD(n_components=50)
svd_vecs_sic4 = svd_sic4.fit_transform(sic4_encodings)
umap_sic4 = UMAP(n_neighbors=5, metric='euclidean')
umap_vecs_sic4 = umap_sic4.fit_transform(svd_vecs_sic4)

# %%
colours = []
for val in pred_ratio.reindex(list(sic4_name_lookup.keys())).fillna(0):
    if val >= 1:
        colours.append('C2')
    elif val > 0:
        colours.append('C1')
    elif val == 0:
        colours.append('gray')

# %%
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(umap_vecs_sic4[:, 0], svd_vecs_sic4[:, 1], 
#             c=pred_ratio.reindex(list(sic4_name_lookup.keys())).fillna(0), 
            alpha=0.9,
            c=colours)
ax.set_xticklabels('')
ax.set_yticklabels('')
plt.savefig(f'{project_dir}/figures/umap_sic4_code_embeddings_scatter.png', **fig_save_opts);

# %%
label_sections = test_df['label'].map(sic4_to_div_lookup)
pred_sections = test_df['pred'].map(sic4_to_div_lookup)

label_section_counts = label_sections.value_counts()
pred_section_counts = pred_sections.value_counts()

# %%
pred_section_ratio = pred_section_counts / label_section_counts
pred_section_ratio.fillna(0, inplace=True)

# %%
fig, ax = plt.subplots(figsize=(6, 6))

colours = ['C0' if v > 1 else 'C3' for v in pred_section_ratio.sort_values()]
pred_section_ratio.sort_values().plot.barh(ax=ax, color=colours)
ax.axvline(1, color='gray', linestyle='--')
ax.set_xlabel('Ratio of Predicted Sections to Labelled Sections');

plt.savefig(f'{project_dir}/figures/predicted_sic_section_representation_barh.png', **fig_save_opts)
# plt.tight_layout()

# %% [markdown]
# We can see that some entire industrial sections are generally over-represented, whereas some do not appear at all. Why is that?

# %%
train_label_counts = train_df['label'].value_counts()
train_test_sic4_index = train_label_counts.index.intersection(pred_ratio.index)

x = train_label_counts.loc[train_test_sic4_index] 
y = pred_ratio.loc[train_test_sic4_index].fillna(0)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x, y, alpha=0.7)

for c, r, i in zip(x, y, x.index):
    if (c > 5000) and (r < 1):
        if i in sic4_name_lookup:
            ax.annotate(sic4_name_lookup[i][:40], (c, r))
    if (c < 3000) and (r > 2.75):
        if i in sic4_name_lookup:
            ax.annotate(sic4_name_lookup[i], (c, r))
    if (c > 5000) and (r > 1):
        if i in sic4_name_lookup:
            ax.annotate(sic4_name_lookup[i], (c, r))
    if (c > 1800) and (r < .25):
        if i in sic4_name_lookup:
            ax.annotate(sic4_name_lookup[i], (c, r))
            
ax.set_xlabel('Frequency in Training Data')
ax.set_ylabel('Prediction Representation')
plt.tight_layout()
plt.savefig(f'{project_dir}/figures/predicted_sic4_representation_vs_training_frequency_scatter.png', **fig_save_opts);

# %% [markdown]
# A higher frequency in the training data doesn't necessarily lead to a higher representation in the predictions. As we would expect from a transformer based language model, the predications are based on more nuanced characteristics than simply predicting based on probability.

# %%
top_n = 20

test_df['max_prob'] = np.max(pred_probs, axis=1)
mean_pred_probs = test_df.groupby('label')['max_prob'].mean().sort_values()

fig, axs = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)

mean_pred_probs.dropna()[-top_n:].plot.barh(ax=axs[0])
mean_pred_probs[:top_n].plot.barh(ax=axs[1])

for ax in axs.ravel():
    y_tick_labels = ax.get_yticklabels()
    for t in y_tick_labels:
        t.set_text(sic4_name_lookup.get(t.get_text()))        
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel('')

ax.set_xlabel('Mean Prediction Probability')

plt.tight_layout()
plt.savefig(f'{project_dir}/figures/mean_sic4_prediction_probability_barh.png', **fig_save_opts)

# %% [markdown]
# The mean prediction probability for the predicted label is an indicator of how uncertain the model is when predicting for that code. Codes with a high average are more easily distinguishable by the classifier, whereas low averages indicat that the probability was more widely distributed across multiple possible labels.

# %%
x = train_label_counts.loc[train_test_sic4_index] 
y = mean_pred_probs.loc[train_test_sic4_index].fillna(0)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x, y, alpha=0.7)
            
ax.set_xlabel('Frequency in Training Data')
ax.set_ylabel('Mean Prediction Probability')
ax.set_xscale('log')

plt.tight_layout()
plt.savefig(f'{project_dir}/figures/mean_sic4_prediction_probability_vs_training_frequency_scatter.png', 
            **fig_save_opts);

# %% [markdown]
# We can see that although there is a tendency for the mean prediction probabilty to increase with the frequency of a label in the training data, there are several codes which buck this trend. Some codes enjoy a relatively high average prediction probability value, despite being one or two orders of magnitude less frequent than the most frequent codes. On the other hand, there are some labels with frequencies in the high thousands which are much less easily classified.
#
# This means that some labels are easily to identify due to a combination of the nature of the training data and the ease with which certain organisations can be identified.

# %%
from scipy.spatial.distance import cdist

# %%
org_ids = test_df.sample(10).index

# %%
closest_sic4_args = np.argsort(cdist(test_encodings[org_ids], sic4_encodings, metric='cosine'), axis=1)[:, :5]
closest_sic4_descriptions = [np.array(list(sic4_name_lookup.values()))[c] for c in closest_sic4_args]

# %%
for i, c in zip(org_ids, closest_sic4_descriptions):
    print(test_df.loc[i]['text'])
    for t in c:
        print('-', t)
    print('')


# %% [markdown]
# ### Distance between company and code descriptions

# %%
def matrix_to_vector_distance(X, y, metric):
    dists = cdist(X, [y], metric=metric)
    return dists


# %%
dist_ids = []
dist_vals = []

for label, group in test_df.groupby('label'):
    ids = group.index
    desc_encodings = test_encodings[ids]
    if label not in sic4_name_lookup:
        continue
    label_id = list(sic4_name_lookup).index(label)
    label_encoding = sic4_encodings[label_id]
    dists = matrix_to_vector_distance(desc_encodings, label_encoding, 'cosine')
    dist_ids.extend(ids)
    dist_vals.extend(dists.ravel())

# %%
test_df['label_description_dist'] = pd.Series(dist_vals, index=dist_ids)

# %%
X = test_df.sort_values('label_description_dist', ascending=False).iloc[10000:10010]

# %%
for t, l in zip(X['text'], X['label']):
    print(sic4_name_lookup[l])
    print(t, '\n')

# %% [markdown]
# ### Cluster coherence (silhouette score)

# %%
from sklearn.metrics import silhouette_samples

# %%
sample_silhouette_values = silhouette_samples(test_encodings, test_df['label'], metric='euclidean')

means = []
for code in sic4_name_lookup.keys():
    ids = test_df[test_df['pred'] == code].index
    means.append(sample_silhouette_values[ids].mean())

# %%
label_silhouettes = pd.Series(means, index=sic4_name_lookup.values()).dropna().sort_values()

# %%
top_n = 10

fig, axs = plt.subplots(nrows=2, figsize=(12, 6), sharex=True)

label_silhouettes[-top_n:].plot.barh(ax=axs[0])
label_silhouettes[:top_n].plot.barh(ax=axs[1])

# for ax in axs.ravel():
#     y_tick_labels = ax.get_yticklabels()
#     for t in y_tick_labels:
#         t.set_text(sic4_name_lookup.get(t.get_text()))        
#     ax.set_yticklabels(y_tick_labels)

axs[1].set_xlabel('Mean Silhouette Score')

plt.tight_layout()
plt.savefig(f'{project_dir}/figures/predicted_sic4_silhouette_barh.png', **fig_save_opts);

# %%
test_df[test_df['pred'].map(sic4_name_lookup) == 'Non-specialised wholesale trade']['text'].sample(10).values

# %% [markdown]
# ### Predicted together but semantically disimilar

# %%
prediction_pairs = Counter([tuple(sorted(v)) for v in np.argsort(pred_probs, axis=1)[:, :2]])

# %%
from scipy.spatial.distance import cosine

# %%
dists = []

for k in prediction_pairs.keys():
    dists.append(cosine(sic4_encodings[k[0]], sic4_encodings[k[1]]))

# %%
prediction_pair_df = pd.DataFrame(
    {'dist': dists, 
     'pair': list(predicion_pairs.keys()), 
     'freq': list(predicion_pairs.values())})

# %%
for pair in prediction_pair_df[(prediction_pair_df['freq'] > 50) & (prediction_pair_df['dist'] > 1)]['pair']:
    print(sic4_name_lookup[le.inverse_transform([pair[0]])[0]])
    print(sic4_name_lookup[le.inverse_transform([pair[1]])[0]])
    print('')


# %% [markdown]
# ### Distance in the taxonomy

# %%
def taxonomy_distance(labels, preds):
    digit_2 = labels.str[:2] == preds.str[:2]
    score_2 = [3 if not d else 0 for d in digit_2]
    digit_3 = labels.str[:3] == preds.str[:3]
    score_3 = [2 if not d else 0 for d in digit_3]
    digit_4 = labels.str[:4] == preds.str[:4]
    score_4 = [1 if not d else 0 for d in digit_4]
    
    scores = np.max(np.array([score_2, score_3, score_4]), axis=0)
    
    return scores


# %%
test_df['taxonomy_dist'] = taxonomy_distance(test_df['label'], test_df['pred'])

# %%
test_df.groupby('label')['taxonomy_dist'].mean().sort_values()[:10]

# %%
list(test_df.groupby('label')['taxonomy_dist'].mean().sort_values()[:10].index.map(sic4_name_lookup))

# %%
list(test_df.groupby('label')['taxonomy_dist'].mean().sort_values()[-10:].index.map(sic4_name_lookup))

# %%

# %%
import networkx as nx
from collections import Counter

# %%
edges = zip(prediction_df['label'], prediction_df['pred'])
edge_weights = [(k[0], k[1], v) for k, v in Counter(tuple(e) for e in edges).items() if v > 10]

# %%
g = nx.Graph()

# %%
g.add_weighted_edges_from(edge_weights)

# %%
nx.draw(g, node_size=3)

# %%
