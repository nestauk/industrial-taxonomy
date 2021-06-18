# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import matplotlib.pyplot as plt
import seaborn as sn
import json
import numpy as np
import pandas as pd
from industrial_taxonomy import project_dir
from industrial_taxonomy.getters.glass import get_organisation_description
from industrial_taxonomy.taxonomy.taxonomy_community import make_network_from_coocc
import altair as alt
from industrial_taxonomy.getters.processing import get_company_sector_lookup, get_sector_name_lookup, get_sector_reassignment_outputs

from industrial_taxonomy.utils.sic_utils import extract_sic_code_description, load_sic_taxonomy, make_sic_lookups, section_code_lookup

from industrial_taxonomy.scripts.extract_communities import get_glass_tokenised

alt.data_transformers.disable_max_rows()

# %%
from industrial_taxonomy.utils.sic_utils import load_sic_taxonomy, section_code_lookup, extract_sic_code_description

# %%
# Read data
company_label_lookup = get_company_sector_lookup()
label_name_lookup = get_sector_name_lookup()

# %%
# Get the sic name / code lookups
section_name_lookup = {k: k+": "+v for k,v in extract_sic_code_description(load_sic_taxonomy(),'SECTION').items()}
division_section_names = {k:section_name_lookup[v.split(" :")[0]] for k,v in section_code_lookup().items()}

# %% [markdown]
# ### Check initial results from the topic modeling

# %%
from scipy.stats import zscore

# %%
# Number of communities per sector

sector_labels = [(k.split("_")[0], int(k.split("_")[1])) for k in label_name_lookup.keys()]
sic_4_sectors = set([x[0] for x in sector_labels])

sector_label_counts = (pd.DataFrame(
    [[v, len([x[1] for x in sector_labels if x[0]==v])] for v in sic_4_sectors],columns=['sic4','communities'])
                       .assign(section=lambda df: df['sic4'].apply(lambda x: x[:2]).map(division_section_names))
                       .assign(nec = lambda df: ['9' in x[2:] for x in df['sic4']])
                      )
sector_label_counts = sector_label_counts.loc[sector_label_counts['communities']>1]

number_base = (alt.Chart(sector_label_counts)
               .encode(y=alt.Y('jitter:Q',title=''),
                        row='nec'))


number_comms_chart = (number_base
                      .transform_calculate(
                          jitter='sqrt(-2*log(random()))*cos(2*PI*random())')
                      .mark_point(filled=True,opacity=0.5)
                      .encode(x=alt.X('communities', title='Number of communities'), 
                        color=alt.Color('section',legend=alt.Legend(columns=2,orient='bottom')), 
                                        tooltip=['sic4']
                       ).properties(height=50,width=400)
               )
number_comms_chart


# %%
# Get minimum description lengths

def get_topic_mdl():
    
    return (pd.read_csv(f"{project_dir}/data/processed/topsbm_mdl.csv",
                       dtype={'sector':str})
            .rename(columns={'model':'mdl'}))


# %%
mdl = get_topic_mdl()

topmodel_output = (sector_label_counts
                   .merge(mdl,left_on='sic4',right_on='sector')
                   .assign(mdl_scaled= lambda df: zscore(df['mdl']))
                   .assign(communities_scaled= lambda df: zscore(df['communities'])))
                   

topmodel_output_chart = (alt.Chart(topmodel_output)
              .mark_point(filled=True,stroke='black',strokeWidth=0.5)
              .encode(x='mdl_scaled',
                     y='communities_scaled',
                     color='section',
                     tooltip=['sector'], 
                     shape=alt.Shape('nec',scale=alt.Scale(range=['circle','square'])))
             )
topmodel_output_chart

# %% [markdown]
# ### Visualise evolution of distance after reassignment

# %%
sector_reassignment_output = get_sector_reassignment_outputs()

# %%
sector_reassignment_output.keys()

# %%
# Line chart: evolution of distances (and confidence intervals)

dist_vect = sector_reassignment_output['distance_container']

dist_summary = [(np.mean(x), np.std(x)) for x in dist_vect]

dist_summary_df = pd.DataFrame(dist_summary, columns=['mean','std'])
dist_summary_df['low'],dist_summary_df['high'] = [dist_summary_df['mean']+1*v*dist_summary_df['std'] for v in [-1,1]]
dist_summary_df['iteration'] = range(len(dist_summary_df))

# %%
dist_base = (alt.Chart(dist_summary_df)
        .encode(x='iteration')
            )

dist_line = (dist_base
             .mark_line(point=True)
             .encode(y=alt.Y('mean',scale=alt.Scale(zero=False))
                    ))
             
dist_error = (dist_base
             .mark_errorbar()
             .encode(y='low',y2='high'))

dist_line


# %%
# Stability: 

# Show share of final reassignmnents

transitions = sector_reassignment_output['transition_container']

def get_transitions(_list):
    
    same_code = _list[0]==_list[1]
    same_sic = _list[0][:4] == _list[1][:4]
    same_division = _list[0][:2] == _list[1][:2]
    
    return same_code, same_sic, same_division


# %%
transition_comparison = [[get_transitions(el) for el in iteration.values()] for iteration in transitions]

# %%
transition_summary = [[np.mean([el[n] for el in org]) for org in transition_comparison] for n in [0,1,2]]

# %%
transition_df = pd.DataFrame(transition_summary).T
transition_df.columns = ['same_community','same_sic_code','same_division']
transition_df['iteration'] = range(len(transition_df))

transition_df_long = transition_df.melt(id_vars='iteration')

transition_chart = (alt.Chart(transition_df_long)
                   .mark_line(point=True)
                   .encode(x='iteration', y=alt.Y('value',
                                                  title='% of reassignments',
                                                  axis=alt.Axis(format='%')), 
                                                  color='variable'))
transition_chart

# %%
from numpy.random import choice

# %%
# Matrix of distances between first and last transition

first_sector = [trans[0] for trans in transitions[0].values()]
last_sector = [trans[1] for trans in transitions[-1].values()]

transition_pairs_df = pd.DataFrame([[x,y] for x,y in zip(first_sector,last_sector)],columns=['initial_sector', 'final_sector'])

# Transitions frequencies

selected = choice(list(set(transition_pairs_df['initial_sector'])),1600, replace=False)

transition_freqs = (transition_pairs_df
                    .groupby('initial_sector')['final_sector']
                    .value_counts(normalize=True)
                    .reset_index(name='share') 
                   )

transition_freqs_filt = transition_freqs.loc[
    (transition_freqs['initial_sector'].isin(selected))&(transition_freqs['final_sector'].isin(selected))]

transition_heatmat = (alt.Chart(transition_freqs_filt)
                    .mark_rect()
                    .encode(y=alt.Y('initial_sector',axis=alt.Axis(labels=False, ticks=False)),
                           x=alt.X('final_sector',axis=alt.Axis(labels=False, ticks=False)),
                           color=alt.Color('share', scale=alt.Scale(type='log')),
                      tooltip=['initial_sector', 'final_sector', 'share'])
                     ).properties(width=400,height=400)

transition_heatmat

# %%
# Transition 4-digit

transition_pairs_df['initial_sic4'],transition_pairs_df['final_sic4'] = [[x[:4] for x in df] for df in [transition_pairs_df['initial_sector'],
                                                                     transition_pairs_df['final_sector']]]

transition_sic_freqs = (transition_pairs_df
                        .groupby('initial_sic4')['final_sic4'].value_counts(normalize=True)
                        .reset_index(name='share')
                       )

sic_name_descr = extract_sic_code_description(load_sic_taxonomy(),'Class')

# %%
transition_sic_freqs['initial_name'],transition_sic_freqs['final_name'] = [transition_sic_freqs[var].map(sic_name_descr) 
                                                                           for var in ['initial_sic4', 'final_sic4']]

transition_heatmat_sic = (alt.Chart(transition_sic_freqs)
                    .mark_rect()
                    .encode(x=alt.X('final_sic4',axis=alt.Axis(labels=False, ticks=False)),
                           y=alt.Y('initial_sic4',axis=alt.Axis(labels=False, ticks=False)),
                           color=alt.Color('share', scale=alt.Scale(type='log')),
                      tooltip=['initial_name', 'final_name', 'share'])
                     ).properties(width=400,height=400)

transition_heatmat_sic

# %%
# Finally 

# What do different sectors get reallocated to?
transition_first_last = [get_transitions([f,l]) for f,l in zip(first_sector,last_sector)]

assessment_cols = ['same_community','same_sic_code','same_division']

transition_assessment_df = (pd.concat([transition_pairs_df,
                                 pd.DataFrame(transition_first_last,
                                             columns=['same_community','same_sic_code','same_division'])],
                                  axis=1)
                         .assign(section = lambda df: df['initial_sector'].apply(lambda x: division_section_names[x[:2]])
                                ))


#selected = choice(list(set(transition_assessment_df['initial_sector'])),1600, replace=False)

transition_shares_df = (transition_assessment_df
                        #.loc[transition_assessment_df['initial_sector'].isin(selected)]
                        .groupby(['initial_sector', 'section'])[assessment_cols]
                        .mean()
                        .reset_index(drop=False)
                        .melt(id_vars=['initial_sector', 'section'])
                        .reset_index(drop=False)
                        .assign(comm_name = lambda df: df['initial_sector'].map(label_name_lookup))
                       )

reassignment_stat_chart = (alt.Chart(transition_shares_df)
                           .mark_point(filled=True, opacity=0.5, stroke='black', strokeWidth=0.2)
                           .encode(y=alt.Y('initial_sector',axis=alt.Axis(labels=False, ticks=False)),
                                   x=alt.X('value', axis=alt.Axis(format='%')),
                                   color='section',
                                   tooltip=['initial_sector', 'comm_name', 'value'],
                                   facet=alt.Facet('variable')
                                  )
                          ).properties(width=200,height=500)

reassignment_stat_chart

# %% [markdown]
# ### Rename communities based on the latest results?
#
#

# %%
from industrial_taxonomy.scripts.extract_communities import get_glass_tokenised
from industrial_taxonomy.taxonomy.taxonomy_filtering import (
    make_doc_term_matrix,
    make_tfidf_mat,
    filter_salient_terms,
    get_promo_terms,
)

# %%
glass_tok = get_glass_tokenised()

# %%
final_community = {k:v[1] for k,v in transitions[-1].items()}

# %%
glass_tok['comm_assigned'] =  glass_tok['org_id'].map(final_community)
glass_tok_filtered = (glass_tok
                      .dropna(axis=0,subset=['comm_assigned'])
                      .reset_index(drop=True)
                     )

# %%
dtm = make_doc_term_matrix(glass_tok_filtered, sector='comm_assigned', tokens='tokens_clean')

# %%
drop_high_freq = dtm.sum().sort_values(ascending=False)[:150].index.tolist()

# %%
tfidf = make_tfidf_mat(dtm.drop(axis=1,labels=drop_high_freq))

# %%
sector_label2_lookup = tfidf.apply(lambda x: ' '.join(x.sort_values(ascending=False).index[:10]),axis=1).to_dict()

# %% [markdown]
# ### Bottom up taxonomy

# %%
from cdlib.algorithms import louvain, sbm_dl_nested
import graph_tool.all as gt
import networkx as nx
from itertools import chain

def make_gt_network(net: nx.Graph) -> list:
    """Converts co-occurrence network to graph-tool netwotk"""
    nodes = {name: n for n, name in enumerate(net.nodes())}
    index_to_name = {v: k for k, v in nodes.items()}
    edges = list(net.edges(data=True))

    g_net = gt.Graph(directed=False)
    g_net.add_vertex(len(net.nodes))

    eprop = g_net.new_edge_property("int")
    g_net.edge_properties["weight"] = eprop

    for edg in edges:
        n1 = nodes[edg[0]]
        n2 = nodes[edg[1]]

        e = g_net.add_edge(g_net.vertex(n1), g_net.vertex(n2))
        g_net.ep["weight"][e] = edg[2]["weight"]

    return g_net, index_to_name


def get_community_names(partition, index_to_name, level=1):
    """Create node - community lookup"""

    b = partition.get_bs()

    b_lookup = {n: b[level][n] for n in sorted(set(b[0]))}

    names = {index_to_name[n]: int(b_lookup[c]) for n, c in enumerate(b[0])}

    return names


# %%
sectors_cooccs, sector_distances = sector_reassignment_output['sector_cooccurrences'],sector_reassignment_output['sector_distances']

all_sectors = list(set(chain(*sectors_cooccs)))

# %%
dist = {s:[] for s in all_sectors}

# %%
# %%time
for n,sect in enumerate(all_sectors):
    if n%100==0:
        logging.info(n)
    for s,d in zip(sectors_cooccs, sector_distances):
        for els,eld in zip(s,d):
            if els == sect:
                dist[sect].append(eld)
            

# %%
dist_stats_dict = {k: (np.mean(v),np.std(v)) for k,v in dist.items()}

dist_stats = pd.DataFrame(dist_stats_dict).T
dist_stats.columns = ['mean','std']

dist_stats['low'],dist_stats['high'] = [dist_stats['mean']+1*v*dist_stats['std'] for v in [-1,1]]
dist_stats = (dist_stats
              .reset_index(drop=False)
              .assign(section = lambda df: df['index'].apply(lambda x: division_section_names[x[:2]])
                     )
             )

# %%
dist_chart = (alt.Chart(dist_stats)
             .mark_point(filled=True, opacity=0.5, stroke='black', strokeWidth=0.2)
             .encode(y=alt.Y('index',axis=alt.Axis(labels=False, ticks=False)),
                     x=alt.X('mean'),
                     color='section')
            ).properties(width=200,height=450)

dist_chart

# %%
p = 0
dist_thres = {k: v[0]-p*v[1] for k,v in dist_stats_dict.items()}

# %%
# %%time
sector_coocc_filtered = []
for s,d in zip(sectors_cooccs, sector_distances):
    occurrences = []
    for els,eld in zip(s,d):
        if eld < dist_thres[els]:
            occurrences.append(els)
    sector_coocc_filtered.append(occurrences)

# %%
sector_coocc_filtered_2 = [x for x in sector_coocc_filtered if len(x)>0]

print(len(sector_coocc_filtered_2))

# %%
net = make_network_from_coocc(sector_coocc_filtered_2, spanning=True, extra_links=500)
g_net,index_name_lu = make_gt_network(net)


# %%
def make_partition(g_net, comms, distr_type):
    logging.info(distr_type)
    state = gt.minimize_nested_blockmodel_dl(g_net, B_min=comms, deg_corr=True,
                                             state_args=dict(recs=[g_net.ep.weight],
                                                                rec_types=[distr_type]))

    state = state.copy(bs=state.get_bs() + [np.zeros(1)] * 4, sampling=True)

    for i in range(100):
        ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)


    return state

# %%
p = make_partition(g_net, 400, 'discrete-poisson')

# %%
# partitions = [make_partition(g_net, 150, distr) for distr in ['discrete-geometric', 
#                                                               'discrete-poisson',
#                                                              # 'discrete-binomial',
#                                                              # 'real-normal'
#                                                              ]]
# [x.entropy() for x in partitions]
# bp = partitions[1]
# bp


# %%
p.draw(
    eorder=g_net.ep.weight
    #edge_pen_width=gt.prop_to_size(g_net.ep.weight, 0.5, 1, power=1)
)

# %%
bs = p.get_bs()

chains = []
chains_named = []

for n,el in enumerate(bs[0]):
    links = [el]
    links_named = [index_name_lu[n]]
    for it in range(1,4):
        new_el = [x for m,x in enumerate(bs[it]) if m==links[-1]]
        links.append(new_el[0])
        links_named.append(f"l{str(it)}_{str(new_el[0])}")

        
    chains.append(links)
    chains_named.append(links_named)

# %%
level_dict = {l[0]:l[1:] for l in chains_named}

# %%
for n in range(0,4):
    level_dict_sub = {k:v[n] for k,v in level_dict.items()}
    glass_tok[f'sector_level_{str(n+1)}'] = glass_tok['comm_assigned'].map(level_dict_sub)
    

# %%
# I want to remove promotional terms



# %%
def make_sector_vocab(glass, sector,tokens='tokens_clean',drop=150, words=10, min_occ=100):
    
    dtm = make_doc_term_matrix(glass, sector=sector, tokens=tokens,min_occurrence=min_occ)
    drop_high_freq = dtm.sum().sort_values(ascending=False)[:drop].index.tolist()
    
    marketing = list(get_promo_terms(dtm))
    
    tfidf = make_tfidf_mat(dtm.drop(axis=1,labels=drop_high_freq+marketing))
    
    label_lu = tfidf.apply(lambda x: ' '.join(x.sort_values(ascending=False).index[:words]),axis=1).to_dict()
    return label_lu
    


# %%
sector_names  = [
    make_sector_vocab(glass_tok, sector=sector_name,
                      tokens='tokens_clean',
                      drop=400, words=20, min_occ=min_occ) for 
    sector_name,min_occ in zip(
        ['comm_assigned','sector_level_1','sector_level_2','sector_level_3'],[50,100,200,400])]

# %%
sector_names_merged = {k:k+": "+v for k,v in {k:v for d in sector_names for k,v in d.items()}.items()}

# %% [markdown]
# ### Visualise sector network
#

# %%
from industrial_taxonomy.altair_network import plot_altair_network
from itertools import combinations

# %%
# Create network => 

sector_col_names = ['comm_assigned','sector_level_1','sector_level_2','sector_level_3']

sector_array = np.array(glass_tok[sector_col_names].dropna())

sector_pairs = list(chain(*[[[x[n],x[n+1]] for x in sector_array] for n in range(3)]))

sector_net = make_network_from_coocc(sector_pairs)


# %%
edges_original = [(e[0],e[1],{"weight":1}) for e in sector_net.edges()]
final_layer = [(c[0],c[1],{'weight':1}) for c in combinations(list(set(sector_array[:,3])),2)]

sector_net_2 = nx.Graph(chain(*[edges_original,final_layer]))

# %%
sect_pos = (pd.DataFrame(
    nx.nx_agraph.graphviz_layout(sector_net_2)
    #nx.kamada_kawai_layout(sector_net_2)
                        )
       .T
       .reset_index(drop=False)
       .rename(columns={0:'x',1:'y','index':'node'})
      )
sector_freqs = pd.concat([glass_tok[s].value_counts()
                          .reset_index(name='sector') for s in sector_col_names]).set_index('index')['sector'].to_dict()

# %%
sector_comms = {el:str(n) for n,comm in enumerate(algorithms.louvain(sector_net).communities) for el in comm}

# %%
# freqs
sector_net_df = (sect_pos
                 .assign(node_size=lambda df: df['node'].map(sector_freqs))
                 .assign(node_color=lambda df: df['node'].map(sector_comms))
                 .assign(node_name=lambda df: df['node'].map(sector_names_merged))
                )

# %%
# Build net

sector_net_chart = plot_altair_network(sector_net_df,
                                      sector_net,
                                       show_neighbours=True,
                                      node_label='node_name',
                                      node_size='node_size',
                                      node_color='node_color',
                                      **{'node_size_title':'freq',
                                         'node_color_title':'comm',
                                         'edge_weight_title':'link',
                                         'title':'text'
                                        })

# Extract positions (radial?)

# Create df and visualise

# %%
sector_net_chart.properties(height=350,width=600)

# %% [markdown]
# ### Exploratory analysis

# %%
import pickle


# %%
def load_topsbm():
    
    with open(f"{project_dir}/models/top_sbm_models.p",'rb') as infile:
        return pickle.load(infile)


# %%
topsbms = load_topsbm()

# %%
sector_6201 = list(mdl['sector']).index('8299')

# %%
topmodel_6201 = topsbms[sector_6201]

topmodel_6201.plot(nedges=3000)

# %%
from industrial_taxonomy.getters.glass import get_organisation_description

descriptions = get_organisation_description()

# %%
# _ids = glass_tok.loc[glass_tok['sector_level_1']=='l1_13']['org_id']
# len(_ids)
# for d in descriptions.loc[descriptions['org_id'].isin(_ids)].drop_duplicates('org_id')['description']:
#     print(d[:1000])
#     print("\n")

# %%
for b in bs:
    print(len(set(b)))


# %%
# test = get_community_names(state,index_name_lu,level=0)

# test_df = (pd.Series(test).reset_index(name='comm_affiliation')
#            .assign(sector_name=lambda df: df['index'].map(se))
#           )
# grouped = test_df.groupby(['comm_affiliation'])['sector_name'].apply(lambda x: "     ".join(list(x)))

# grouped[0]

# %%
# Build and analyse network

#sector_quantile = comp_sector_sim_df_2.quantile(0.9)

#Function that loops over rows and compares 

def get_threshold(vector,scale=2):
    
    return np.mean(vector)+scale*np.std(vector)

def get_co_occurrences(sims, thres_series):
    
    comparison = sims >thres_series
    return comparison.loc[comparison==True].index.tolist()

def make_gt_network(net: nx.Graph) -> list:
    """Converts co-occurrence network to graph-tool netwotk"""
    nodes = {name: n for n, name in enumerate(net.nodes())}
    index_to_name = {v: k for k, v in nodes.items()}
    edges = list(net.edges(data=True))

    g_net = gt.Graph(directed=False)
    g_net.add_vertex(len(net.nodes))

    eprop = g_net.new_edge_property("int")
    g_net.edge_properties["weight"] = eprop

    for edg in edges:
        n1 = nodes[edg[0]]
        n2 = nodes[edg[1]]

        e = g_net.add_edge(g_net.vertex(n1), g_net.vertex(n2))
        g_net.ep["weight"][e] = edg[2]["weight"]

    return g_net, index_to_name


def get_community_names(partition, index_to_name, level=1):
    """Create node - community lookup"""

    b = partition.get_bs()

    b_lookup = {n: b[level][n] for n in sorted(set(b[0]))}

    names = {index_to_name[n]: int(b_lookup[c]) for n, c in enumerate(b[0])}

    return names



# %%
thres_vect = pd.Series([get_threshold(sect, scale=2) for col,sect in comp_sector_sim_df_2.iteritems()],
                      index=comp_sector_sim_df_2.columns)


# %%
sector_co_occs = [get_co_occurrences(row,thres_vect) for _,row in comp_sector_sim_df_2.iterrows()]

# %%
sector_co_occs_connection = [x for x in sector_co_occs if len(x)>0]

# %%
net = make_network_from_coocc(sector_co_occs_connection, spanning=True)

# %%
g_net,index_name_lu = make_gt_network(net)

# %%
state = gt.minimize_nested_blockmodel_dl(g_net, B_min=250)

# %%
state.draw()

# %%
test = get_community_names(state,index_name_lu,level=0)

test_df = (pd.Series(test).reset_index(name='comm_affiliation')
           .assign(sector_name=lambda df: df['index'].map(label_name_lookup))
          )
grouped = test_df.groupby(['comm_affiliation'])['sector_name'].apply(lambda x: " ".join(list(x)))

print(len(grouped))

# %%
# for k in grouped.keys():
    
#     print(k)
#     print(grouped[k][:1000])
#     print("\n")


# %%
#state.get_bs()[0]

# %%
#len(set(state.get_bs()[0]))

# %%
state.entropy()

# %%
gt.
