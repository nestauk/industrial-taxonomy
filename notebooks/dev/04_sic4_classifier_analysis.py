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

# %%
# %run ../notebook_preamble.ipy

# %%
import joblib

# from umap import UMAP
from sklearn.decomposition import TruncatedSVD

from industrial_taxonomy.sic import make_sic_lookups

import seaborn as sns

import altair as alt

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
# - [ ] How challenging was it for the model to classify certain companies or codes?
#     - [x] What true codes have the widest distribution of predicted codes?
# - [ ] How coherent/homogeneous are the companies within a certain part of the taxonomy?
#     - [x] Calculate the silhouette score for each group of companies based on their true SIC code.
# - [ ] What is the semantic relationship between codes and descriptions?
#     - [ ] How far are company descriptions that are wrongly labelled to the ones that are truly labelled?
#     - [ ] Between the codes of incorrectly labelled?
#     - [ ] Between the codes and descriptions of all labelled?
# - [ ] Look at the average semantic similarity between the group that was predicted and the group that is the true label 
# - [ ] Limitations of the current taxonomy
#     - [ ] Why are these codes difficult to classify? Is it the company descriptions or the limitations of the codes?
#     
# ### Bonus
# - [ ] How noisy is the labelling in CH?
#     - [ ] What is the semantic distance between company descriptions and their CH labels?
# - [ ] What is the qualitative difference between the predicted and true SIC code where they differ?
#     - [ ] Are they in different parts of the taxonomy?
#     - [ ] Are they more specific/general? (when they are just 1 move away in the taxonomy)
# - [ ] How do we reverse engineer some of the sources of error to find out where they have happened?
#     - [ ] Train a text generation or augmentation algorithm to more succesfully predict the true label? How much does this language differ?
# - [ ] What are the limitations of the current codes for describing the companies?
#   - [ ] Codes that are predicted together but are semantically disimilar.
#     
#     
#

# %%
from metaflow import Run, Metaflow

# %%
from industrial_taxonomy import config

# %%
preproc_run_id = config['flows']['sic4_classifier']['preproc']['run_id']
preproc_run = Run(f'SicPreprocess/{preproc_run_id}')

train_run_id = config['flows']['sic4_classifier']['train']['run_id']
train_run = Run(f'TextClassifier/{train_run_id}')

predict_run_id = config['flows']['sic4_classifier']['predict']['run_id']
predict_run = Run(f'TextClassifierPredict/{predict_run_id}')

embed_run_id = config['flows']['sic4_embedder']['run_id']
embed_run = Run(f'Embedder/{embed_run_id}')

# %%
from collections import Counter
from sklearn.metrics import classification_report

# %%
test_dataset = predict_run.data.test_dataset
test_preds = predict_run.data.pred_labels

label_lookup = preproc_run.data.label_lookup
inv_label_lookup = {v: k for k, v in label_lookup.items()}

# %%
test_labels = [inv_label_lookup[i.label] for i in test_dataset.samples]
pred_labels = [inv_label_lookup[i] for i in test_preds]

# %% [markdown]
# ## 1. Classification Report

# %%
clf_report_df = pd.DataFrame(classification_report(test_labels, pred_labels, output_dict=True)).T
# clf_report_df = clf_report_df[clf_report_df['f1-score'] > 0]
clf_report_df = clf_report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
clf_report_df = clf_report_df.sort_values('f1-score', ascending=False)
clf_report_df['Name'] = clf_report_df.index.map(sic4_name_lookup)
clf_report_df = clf_report_df.drop('9999')

# %%
clf_report_df['SIC'] = (clf_report_df.index
                              + ': ' + clf_report_df['Name'])

# %%
sic4_name_lookup['9609']

# %%
alt.Chart(clf_report_df.head(20)).mark_bar().encode(
    x='f1-score',
    y=alt.Y('SIC', sort='-x'),
#     tooltip=['4-Digit SIC', 'Description'],
#     color=alt.Color('SIC Division', scale=alt.Scale(scheme='category20'))
).interactive().properties(
    width=600,
    height=400
).configure_axis(
    labelFontSize=14,
    titleFontSize=14
)

# %%
clf_report_df.tail(10)

# %% [markdown]
# There is no obvious categorisation of the 4-digit codes based on their classification.

# %% [markdown]
# ### 1.2 Shannon Diversity of Predicted Labels

# %%
from sklearn.metrics import confusion_matrix

# %%
confusion_df = pd.DataFrame(confusion_matrix(test_labels, pred_labels, labels=list(sic4_name_lookup.keys())))
confusion_df.index = list(sic4_name_lookup.keys())
confusion_df.columns = list(sic4_name_lookup.keys())

# %%
from skbio.diversity.alpha import shannon

# %%
true_shannon_scores = pd.DataFrame(confusion_df.fillna(0).apply(shannon, axis=1).sort_values(ascending=False))
true_shannon_scores['Description'] = true_shannon_scores.index.map(sic4_name_lookup)

# %%
true_shannon_scores = true_shannon_scores.rename(columns={0: 'Shannon'})

# %%
true_shannon_scores.head(10)

# %%
true_shannon_scores.dropna().tail(10)


# %% [markdown]
# We find that codes with a high Shannon diversity tend to be those that are very catch-all sectors. On the other hand, sectors with a low Shannon score tend to be more specific.

# %%
# pred_shannon_scores = pd.DataFrame(confusion_df.apply(shannon, axis=1).sort_values(ascending=False))
# pred_shannon_scores['Description'] = pred_shannon_scores.index.map(sic4_name_lookup)

# %%
def get_test_ilocs(code):
    ilocs = []
    for i, sample in enumerate(test_dataset.samples):
        if inv_label_lookup[sample.label] == code:
            ilocs.append(i)
    return ilocs

def get_test_embeddings():
    embeddings = embed_run.data.encodings

    org_data = pd.DataFrame.from_records(preproc_run.data.org_data)
    org_data = org_data.set_index('index')

    test_doc_ids = [s.index for s in test_dataset.samples]

    test_row_ids = []
    for i in test_doc_ids:
        test_row_ids.append(org_data.index.get_loc(i))
        
    test_embeddings = embeddings[test_row_ids]
    test_embeddings = pd.DataFrame(data=test_embeddings, index=test_row_ids)
    return test_embeddings

def get_train_embeddings():
    embeddings = embed_run.data.encodings
    
    org_data = pd.DataFrame.from_records(preproc_run.data.org_data)
    org_data = org_data.set_index('index')
    
    train_doc_ids = [s.index for s in train_dataset.samples]

    train_row_ids = []
    for i in train_doc_ids:
        train_row_ids.append(org_data.index.get_loc(i))
        
    train_embeddings = embeddings[train_row_ids]
    train_embeddings = pd.DataFrame(data=train_embeddings, index=train_row_ids)
    return train_embeddings

# def get_eval_embeddings():


# %% [markdown]
# ### 1.3 Silhouette Scores

# %%
from sklearn.metrics import silhouette_samples, pairwise_distances

# %%
test_embeddings = get_test_embeddings()
test_embeddings_pairwise_cosine = pairwise_distances(test_embeddings, metric='cosine')

# %%
test_division_labels = [sic4_to_div_lookup.get(t, np.nan) for t in test_labels]

# %%
test_silhouette_section = silhouette_samples(
    test_embeddings_pairwise_cosine, 
    labels=test_division_labels,
    metric='precomputed'
)

# %%
test_silhouette_sample_score = silhouette_samples(
    test_embeddings_pairwise_cosine,
    labels=test_labels,
    metric='precomputed'
)

# %%
mean_silhouette_scores = pd.DataFrame(
    {'Label': test_labels, 
     'Silhouette Score': test_silhouette_sample_score}
).groupby('Label').mean()

test_agg_df = true_shannon_scores.join(mean_silhouette_scores)
test_agg_df['SIC Division'] = test_agg_df.index.map(sic4_to_div_lookup)
test_agg_df['4-Digit SIC'] = test_agg_df.index

# %%
alt.Chart(test_agg_df).mark_circle(size=60).encode(
    x='Silhouette Score',
    y='Shannon',
    tooltip=['4-Digit SIC', 'Description'],
#     color=alt.Color('SIC Division', scale=alt.Scale(scheme='category20'))
).interactive().properties(
    width=600,
    height=400
).configure_axis(
    labelFontSize=14,
    titleFontSize=14
)

# %% [markdown]
# We can see that there is a negative correlation between the mean sample Silhouette score of the company description embeddings and the Shannon scores of their predicted labels. This suggests either that sectors with more dispersed company descriptions tend to become resolved into alternative, more appropriate sectors, or that the model is just very poor at identifying those sectors. Or that these sectors are growing because they are absorbing companies from other sectors that are poorly defined or have a large overlap.

# %%
test_agg_df = test_agg_df.join(clf_report_df)
test_agg_df = test_agg_df.rename(columns={'precision': 'Precision', 'recall': 'Recall'})

# %%
alt.Chart(test_agg_df).mark_circle(size=60).encode(
    x='Precision',
    y='Recall',
    tooltip=['4-Digit SIC', 'Description'],
#     size='Silhouette Score'
#     color=alt.Color('SIC Division', scale=alt.Scale(scheme='category20'))
).interactive().properties(
    width=600,
    height=400
).configure_axis(
    labelFontSize=14,
    titleFontSize=14
)

# %% [markdown]
# - 6910 Legal activities - high precision, high recall
# - 9001 Performing arts - high recall, low precision
# - 6820 Renting or operating of own or leased real estate - low precision, low recall
# - 1105 Manufacture of beer - low precision, high recall

# %%
import faiss
from itertools import chain

# %%
index = faiss.IndexFlatIP(test_embeddings.to_numpy().shape[1])
index.add(test_embeddings.to_numpy())
nns = index.search(test_embeddings.to_numpy(), k=2)[1][:, 1:]

# %%
nns_labels = []

for nn in nns:
    l = []
    for n in nn:
        l.append(test_labels[n])
    nns_labels.append(l)
nns_labels = np.array(nns_labels)

# %%
c = Counter(chain(*nns_labels[get_test_ilocs('6910')]))
s = sum(c.values())
for x in c.most_common(5):
    print(str(np.round((x[1] / s) * 100, 1)) + '% -', x[0] + ':', sic4_name_lookup[x[0]])

# %%
c = Counter(chain(*nns_labels[get_test_ilocs('8121')]))
s = sum(c.values())
for x in c.most_common(5):
    print(str(np.round((x[1] / s) * 100, 1)) + '% -', x[0] + ':', sic4_name_lookup[x[0]])

# %%
c = Counter(chain(*nns_labels[get_test_ilocs('6820')]))
s = sum(c.values())
for x in c.most_common(5):
    print(str(np.round((x[1] / s) * 100, 1)) + '% -', x[0] + ':', sic4_name_lookup[x[0]])

# %%
c = Counter(chain(*nns_labels[get_test_ilocs('4753')]))
s = sum(c.values())
for x in c.most_common(6):
    if x[0] in sic4_name_lookup:
        print(str(np.round((x[1] / s) * 100, 1)) + '% -', x[0] + ':', sic4_name_lookup[x[0]])

# %%
from umap import UMAP

# %%
from sklearn.decomposition import TruncatedSVD

# %%
svd = TruncatedSVD(n_components=50)
svd_vecs = svd.fit_transform(test_embeddings)

# %%
umap = UMAP()
umap_vecs = umap.fit_transform(svd_vecs)

# %%
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(
    umap_vecs[:, 0], 
    umap_vecs[:, 1],
    alpha=0.1,
    label='All other codes'
)
ax.scatter(
    umap_vecs[:, 0][[True if l == '8299' else False for l in test_labels]],
    umap_vecs[:, 1][[True if l == '8299' else False for l in test_labels]],
    alpha=0.3,
    label='8299: Other business support service activities n.e.c.'
           )
ax.axis('off')
ax.legend()

# %%
Counter(test_labels).most_common(10)

# %%
code = '7490'

fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(
    umap_vecs[:, 0], 
    umap_vecs[:, 1],
    alpha=0.1,
    label='All other codes'
)
ax.scatter(
    umap_vecs[:, 0][[True if l == code else False for l in test_labels]],
    umap_vecs[:, 1][[True if l == code else False for l in test_labels]],
    alpha=0.3,
    label= code + f': {sic4_name_lookup[code]}'
           )
ax.axis('off')
ax.legend(loc='upper left')

# %%
pred_silhouette_sample_score = silhouette_samples(
    test_embeddings_pairwise_cosine,
    labels=pred_labels,
    metric='precomputed'
)

# %%
mean_pred_silhouette_scores = pd.DataFrame(
    {'Label': pred_labels, 
     'Silhouette Score': pred_silhouette_sample_score}
).groupby('Label').mean()


# %%
prediction_probs = predict_run.data.pred_probs

# %%
prediction_probs[get_test_ilocs(code)].shape

# %%
code = '1105'

max_probs = np.max(prediction_probs[get_test_ilocs(code)], axis=1)
print(np.mean(max_probs[np.array(test_labels)[get_test_ilocs(code)] == np.array(pred_labels)[get_test_ilocs(code)]]))
print(np.mean(max_probs[np.array(test_labels)[get_test_ilocs(code)] != np.array(pred_labels)[get_test_ilocs(code)]]))

# %%
code = '6910'

max_probs = np.max(prediction_probs[get_test_ilocs(code)], axis=1)
plt.hist(max_probs[np.array(test_labels)[get_test_ilocs(code)] == np.array(pred_labels)[get_test_ilocs(code)]], bins=20, alpha=.5)
plt.hist(max_probs[np.array(test_labels)[get_test_ilocs(code)] != np.array(pred_labels)[get_test_ilocs(code)]], bins=20, alpha=.5);

# %%
g = sns.clustermap(pairwise_distances(confusion_df, metric='cosine'))

# %%
g.dendrogram_col.linkage

# %%
get_test_ilocs(code)

# %%
np.array(test_labels)[get_test_ilocs('9001')].shape

# %%
pred_silhouette_sample_score = silhouette_samples(
    test_embeddings_pairwise_cosine,
    labels=pred_labels,
    metric='precomputed'
)

# %%
mean_pred_silhouette_scores = pd.DataFrame(
    {'Label': pred_labels, 
     'Silhouette Score': pred_silhouette_sample_score}
).groupby('Label').mean()


# %%
prob_shannons = [shannon(r) for r in prediction_probs]

# %%
test_records = []

for sample in test_dataset.samples:
    test_records.append({
        'org_id': sample.index,
        'label': inv_label_lookup[sample.label],
        'text': sample.text
        
    })
test_df = pd.DataFrame.from_records(test_records)
del test_records

# %%
test_df['Shannon Prediction Probability'] = prob_shannons

# %%
test_df['pred'] = pred_labels

# %%
for i, row in test_df.sort_values('Shannon Prediction Probability').tail(30).iterrows():
    if row['label'] in sic4_name_lookup:
        print(row['label'], sic4_name_lookup[row['label']])
        print(row['pred'], sic4_name_lookup[row['pred']])
        print(row['text'])
        print('')

# %%
prob_shannons

# %%
pd.DataFrame(data={'Probability Shannon': prob_shannons,
                   'Label': test_labels
                  }).groupby('Label').mean().sort_values('Probability Shannon')

# %%
plt.hist(prob_shannons, bins=100);

# %%
plt.hist(np.max(prediction_probs, axis=1), bins=100);

# %%
test_agg_df['Predicted Silhouette Score'] = mean_pred_silhouette_scores

# %%
s = test_agg_df[['Silhouette Score', 'Predicted Silhouette Score']].dropna().melt(ignore_index=False)
s = s.reset_index()
s = s.rename(columns={'variable': 'Silhouette', 'value': 'Score', 'index': 'SIC Code'})
s['Silhouette'] = s['Silhouette'].map({'Silhouette Score': 'True', 'Predicted Silhouette Score': 'Predicted'})
s = s.sort_values('Silhouette')

# %%
s

# %%
alt.Chart(s).mark_area(
    opacity=0.5,
    interpolate='step'
).encode(
    alt.X('Score', bin=alt.Bin(maxbins=100)),
    alt.Y('count()', stack=None),
    alt.Color('Silhouette:N')
).properties(
#     title='Overlapping Histograms from Tidy/Long Data'
)

# %%
alt.Chart(test_agg_df).mark_bar().encode(
    alt.X("Silhouette Diff:Q", bin=alt.Bin(step=0.025)),
    y='count()',
)

# %%
ids_1105 = get_test_ilocs('6910')

# %%
plt.hist(test_silhouette_sample_score[ids_1105], alpha=0.5, bins=20)
plt.hist(pred_silhouette_sample_score[ids_1105], alpha=0.5, bins=20);

# %% [markdown]
# We can see that the difference between the silhouette scores of the companies grouped by their predicted labels is higher than those where the companies are grouped by 

# %%
plt.scatter(test_agg_df['Silhouette Score'], )

# %%
test_agg_df.reset_index()[['Silhouette Score', 'Predicted Silhouette Score']].unstack

# %%
test_agg_df.reset_index()[['Silhouette Score', 'Predicted Silhouette Score']].melt

# %%
s

# %%
s = test_agg_df[['Silhouette Score', 'Predicted Silhouette Score']].dropna().melt(ignore_index=False)
s = s.reset_index()
s = s.rename(columns={'variable': 'Silhouette', 'value': 'Score', 'index': 'SIC Code'})
s['Silhouette'] = s['Silhouette'].map({'Silhouette Score': 'True', 'Predicted Silhouette Score': 'Predicted'})
s = s.sort_values('Silhouette')


alt.Chart(s).mark_circle(size=60).encode(
    x=alt.Y('SIC Code', sort=alt.EncodingSortField('Score')),
    y='Score',
#     tooltip=['4-Digit SIC', 'Description'],
    color=alt.Color('Silhouette', 
#                     scale=alt.Scale(scheme='category20')
                   )
).interactive()

# %%

# %%
alt.Chart(test_agg_df).mark_circle(size=60).encode(
    x='Silhouette Score',
    y='f1-score',
    tooltip=['4-Digit SIC', 'Description'],
#     color=alt.Color('SIC Division', scale=alt.Scale(scheme='category20'))
).interactive()

# %%
test_agg_df['Silhouette Difference'] = test_agg_df['Predicted Silhouette Score'] - test_agg_df['Silhouette Score']

# %%
alt.Chart(test_agg_df).mark_circle(size=60).encode(
    x='Silhouette Score',
    y='precision',
    tooltip=['4-Digit SIC', 'Description'],
#     color=alt.Color('SIC Division', scale=alt.Scale(scheme='category20'))
).interactive()

# %%
train_dataset = train_run.data.train_dataset
train_labels = [inv_label_lookup[i.label] for i in train_dataset.samples]

# %%
test_agg_df['Training Frequency'] = pd.Series(train_labels).value_counts()
test_agg_df['Prediction Frequency'] = pd.Series(pred_labels).value_counts()
test_agg_df['Test Frequency'] = pd.Series(test_labels).value_counts()

# %%
test_df['pred'] = pred_labels


# %%
def confusion_matrix_sic(code):
    df = test_df[(test_df['pred'] == code) | (test_df['label'] == code)]
    cm = confusion_matrix(df['label'], df['pred'])
    return cm


# %%
def 

test_df[(test_df['pred'] == '9001') | (test_df['label'] == '9001')]

# %%
edges = zip()

# %%
sic4_name_lookup['9001']

# %%
alt.Chart(test_agg_df.reset_index().dropna().sort_values('Predicted Silhouette Score')).mark_bar().encode(
    x2='Silhouette Score',
    x='Predicted Silhouette Score',
    y=alt.Y('index', sort='-x'),
    tooltip=['4-Digit SIC', 'Description'],
#     color=alt.Color('SIC Division', scale=alt.Scale(scheme='category20'))
).interactive()

# %%
test_agg_df

# %%
test_agg_df['Training Frequency'] = 

# %%

# %%
mean_silhouette_scores = pd.DataFrame({'Label': pred_labels, 'Silhouette Score': s}).groupby('Label').mean()

pred_agg_df = true_shannon_scores.join(mean_silhouette_scores)
pred_agg_df['SIC Division'] = pred_agg_df.index.map(sic4_to_div_lookup)
pred_agg_df['4-Digit SIC'] = pred_agg_df.index

# %%
mean_silhouette_scores.sort_values('Silhouette Score')

# %%
test_embeddings.shape

# %%
len(test_row_ids)

# %%
org_data[0]

# %%
embed_run.data


# %%
def print_fn_misclassifications(code, cm, topn=5):
    print('True:', code, '-', sic4_name_lookup[code])
    tops = cm.loc[code].sort_values(ascending=False)[:topn]
    for t in tops.index:
        print(t, '-', sic4_name_lookup[t])


# %%
for code in clf_report_df.tail(5).index:
    print_fn_misclassifications(code, confusion_df)
    print('')

# %%
from collections import defaultdict

# %%
code = defaultdict(list)

for sample in test_dataset.samples:
    label = inv_label_lookup[sample.label]
    if label in clf_report_df.tail(5).index:
        code[label].append(sample)


# %%
train_dataset = train_run.data.train_dataset

# %%
train_label_counts = pd.Series([l.label for l in train_dataset.samples]).value_counts()
pred_label_counts = pd.Series(test_preds).value_counts()

# %%
representation_df = pd.DataFrame({
    'Representation': representation, 
    'Training Frequency': train_label_counts})
representation_df = representation_df.reset_index()
representation_df['4-Digit Code'] = representation_df['index'].map(inv_label_lookup)
representation_df['Sector Name'] = representation_df['4-Digit Code'].map(sic4_name_lookup)
representation_df.drop('index', axis=1, inplace=True)
representation_df['color'] = representation_df['4-Digit Code'].map(sic4_to_div_lookup)
representation_df.dropna(inplace=True)

# %%
representation_df

# %%
alt.Chart(representation_df).mark_circle(size=60).encode(
    x='Training Frequency',
    y='Representation',
    tooltip=['4-Digit Code', 'Sector Name'],
    color=alt.Color('color', scale=alt.Scale(scheme='category20'))
).interactive()

# %% [markdown]
# Sectors to investigate:
#
# - 3299 Other manufacturing n.e.c. - highly over-represented
# - 9001 Performing arts - correctly represented with low training frequency
# - 9606 Other personal services and activities n.e.c. - low representation but with high training frequency
# - 6201 Computer programming activities - low frequency but with high training frequency
# - 5811 Wholesale of beverages - Low representation and low frequency

# %%
import networkx as nx

# %%
g = nx.from_numpy_matrix(confusion_matrix(test_labels, pred_labels))

# %%
g = nx.maximum_spanning_tree(g)

# %%
nx.draw(g, node_size=1)

# %%
representation = (pred_label_counts / train_label_counts).fillna(0)

# %%
plt.scatter(train_label_counts, representation)

# %%
alt.Chart(clf_report_df).mark_circle(size=60).encode(
    x='support',
    y='f1-score',
#     color='Origin',
#     tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
)
# ).interactive()


# %%
i = 50001

print('Text:', test_dataset.samples[i].text)
print('Label:', sic4_name_lookup[inv_label_lookup[test_dataset.samples[i].label]])
print('Predicted:', sic4_name_lookup[inv_label_lookup[test_preds[i]]])

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
