import logging
import os
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from gensim.models import Doc2Vec
import industrial_taxonomy

project_dir = industrial_taxonomy.project_dir


def save_model(mod, name):
    with open(os.path.join(project_dir, f"models/{name}.p"), "wb") as outfile:
        pickle.dump(mod, outfile)


def load_model(name):
    with open(os.path.join(project_dir, f"models/{name}.p"), "rb") as infile:
        model = pickle.load(infile)
    return model


def save_table(table, name,index=False):
    table.to_csv(f"{project_dir}/data/processed/{name}.csv", index=index)


def make_co_occurrence_table(co_occurrence_dict):
    table = pd.concat(
        [
            pd.DataFrame({"sector": [s for s in vals], "company": [k for s in vals]})
            for k, vals in co_occurrence_dict.items()
        ]
    )
    table["value"] = 1
    table_wide = table.pivot_table(
        index="company", columns="sector", values="value"
    ).fillna(0)
    return table_wide


def make_distance_matrix(table, metric="cosine", similarity=False, return_df=True):
    spatial_mat = pairwise_distances(table, metric=metric)
    if similarity is True:
        spatial_mat = 1 - spatial_mat
    if return_df is True:
        spatial_df = pd.DataFrame(spatial_mat, index=table.index, columns=table.index)
        return spatial_df
    else:
        return spatial_mat


def find_duplicated_communities(sim_mat, comm_names, threshold=0.7):

    comm_names_2 = {k: k + ": " + v for k, v in comm_names.items()}

    high_simm_pairs = (
        sim_mat.applymap(lambda x: x if x > threshold else 0)
        .reset_index(drop=False)
        .melt(id_vars="sector", var_name="sector_2")
        .query("value>0")
        .query("sector!=sector_2")
        .reset_index(drop=True)
        .assign(s1_name=lambda df: df["sector"].map(comm_names_2))
        .assign(s2_name=lambda df: df["sector_2"].map(comm_names_2))
        .dropna()
    )
    high_simm_pairs["merged_name"] = [
        f"merged: {r.sector}__{r.sector_2}" for _, r in high_simm_pairs.iterrows()
    ]

    high_simm_pairs["merged_name_terms"] = [
        comm_names[r["sector"]] + "__" + comm_names[r["sector_2"]]
        for _, r in high_simm_pairs.iterrows()
    ]

    merged_communities = high_simm_pairs.set_index("sector")["merged_name"].to_dict()
    merged_communities_terms = high_simm_pairs.set_index("merged_name")[
        "merged_name_terms"
    ].to_dict()

    # Make names
    comm_names_2 = {k: v for k, v in comm_names.items() if k in sim_mat.columns}

    comm_names_merged = make_comm_names_merged(
        comm_names_2, merged_communities, merged_communities_terms
    )

    return merged_communities, comm_names_merged


def make_comm_names_merged(comm_names, merged_communities, merged_communities_terms):
    comm_names_merged = {}
    merged_cont = []
    for k, v in comm_names.items():
        if k not in merged_communities.keys():
            comm_names_merged[k] = v
        elif any(k in mt for mt in merged_cont) is True:
            pass
        else:
            merged_name = merged_communities[k]
            comm_names_merged[merged_name] = merged_communities_terms[merged_name]
            merged_cont.append(merged_name)
    return comm_names_merged


def make_labelled_set(glass_labellled, merged_comms, count_thres):
    glass_labellled["label_2"] = [
        x if x not in merged_comms.keys() else merged_comms[x]
        for x in glass_labellled["label"]
    ]

    comm_counts = glass_labellled["label_2"].value_counts()

    comm_keep = comm_counts.loc[comm_counts > count_thres].index.tolist()
    comm_keep.remove("unlabelled")

    labelled_set = (
        glass_labellled.loc[glass_labellled["label_2"].isin(comm_keep)]
        .groupby("label_2")["org_id"]
        .apply(lambda x: set(x))
    )
    return labelled_set


def train_d2v(tagged_docs, min_count=2,dimensions=100):
    d2v = Doc2Vec(min_count=min_count, vector_size=dimensions)
    d2v.build_vocab(tagged_docs)
    d2v.train(tagged_docs, total_examples=len(tagged_docs), epochs=10)
    return d2v


def make_average_docvecs(labelled_set, doc2v):

    vector_container = []

    for sect in sorted(labelled_set.keys()):
        sect_cont = []
        for _id in labelled_set[sect]:
            vect = doc2v.docvecs[_id]
            sect_cont.append(vect)
        sect_mean = pd.DataFrame(sect_cont).mean()
        sect_mean.shape
        vector_container.append(sect_mean)

    vector_df = pd.DataFrame(vector_container, index=sorted(labelled_set.keys()))
    return vector_df


def find_closest_sector(vect, all_vectors, d2v, counter, token_lim=4):
    if counter % 5000 == 0:
        logging.info(counter)

    if len(vect["token_filtered"]) < token_lim:
        return "unlabelled", np.nan

    my_vect = d2v.docvecs[vect["org_id"]]

    result_test = []

    for _, r in all_vectors.iterrows():

        sim = 1 - cosine(my_vect, r)
        result_test.append([_, sim])

    results = sorted(result_test, key=lambda x: x[1], reverse=True)[0]

    return results[0], results[1]
