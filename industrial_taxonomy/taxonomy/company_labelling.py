# Functions to label companies with community tokens

import pickle
import gensim
import json
import pandas as pd
import numpy as np
from itertools import combinations
from industrial_taxonomy.taxonomy.taxonomy_filtering import (
    make_doc_term_matrix,
    make_tfidf_mat,
)
import industrial_taxonomy

project_dir = industrial_taxonomy.project_dir
MODEL_PATH = f"{project_dir}/models/w2v.p"


def save_model(name):
    with open(MODEL_PATH + f"/{name}.p", "rb") as infile:
        model = pickle.load(infile)
    return model


def get_communities():
    """Load and process sector community dict"""

    with open(f"{project_dir}/data/processed/sector_communities.json", "r") as infile:
        comms = json.load(infile)

    comms = {k: set(v) for k, v in comms.items()}
    return comms


def tag_companies(
    glass_sector: pd.DataFrame,
    description_lu: dict,
    communities: dict,
    exclusive: bool = True,
) -> pd.DataFrame:
    """Tag companies with their related communities
    Args:
        glass_sector: glass df with company descriptions
        description_lookup: lookup between org ids and company descriptions
        communities: lookup between communities and their constituent tokens
        exclusive: do we want to tag every company in one community
    Returns:
        A table with top community for a company or the communities present in the company
    """

    tag_counts = [
        [len(set(tok_filt) & comm) for comm in communities.values()]
        for tok_filt in glass_sector["token_filtered"]
    ]
    tag_counts_df = pd.DataFrame(tag_counts, columns=list(communities.keys()))

    if exclusive is True:
        top_comms_df = (
            pd.concat(
                [
                    tag_counts_df.idxmax(axis=1),
                    tag_counts_df.max(axis=1),
                    glass_sector[["org_id"]],
                ],
                axis=1,
            )
            .rename(columns={0: "top_community", 1: "overlap"})
            .assign(descr=lambda df: df["org_id"].map(description_lu))
        )

        return top_comms_df
    else:
        tag_counts_df.index = glass_sector["org_id"]
        return tag_counts_df


def post_filter(tagged: pd.DataFrame, threshold: int) -> list:
    """Remove small industries from a labelled dataset
    Args:
        tagged: tagged df
        threshold: min size for inclusion
    Returns:
        A list of larger sectors
    """
    sector_counts = tagged["label"].value_counts()
    sector_big = sector_counts.loc[sector_counts > threshold].index.tolist()

    label_post = [x if x in sector_big else "unlabelled" for x in tagged["label"]]
    return label_post


def label_row(row: pd.Series, threshold: int) -> str:
    """Labels a company with a community if it is above a threshold
    Args:
        row: vector with information about a company
        threshold: minimum value to consider that a company is related to a community
    Returns:
        A community or unlabelled label
    """

    return row["top_community"] if row["overlap"] > threshold else "unlabelled"


def make_labelled_dataset(
    glass_tagged: pd.DataFrame, occ_threshold: int
) -> pd.DataFrame:
    """Makes a tagged dataset
    Args:
        glass_tagged: a table of companies with top community and occurrences
        occ_threshold: occurrence threshold for considering a company relates to a community
        sector_size: minimum community size for considering in the analysis
    Returns:
        A labelled dataset
    """

    gt = glass_tagged.copy()

    gt["label"] = gt.apply(lambda x: label_row(x, occ_threshold), axis=1)
    # gt["label_2"] = post_filter(gt, sector_size)
    return gt


# def format_dataset(df):
#     df_ = df.drop(axis=1, labels=["label", "overlap", "top_community"])
#     df_ = df_.rename(columns={"descr": "text", "label_2": "label", "org_id": "index"})
#     return df_.to_dict(orient="records")


def get_salient_terms_community(
    labelled: pd.DataFrame, glass_sector: pd.DataFrame, top_terms: int = 10
) -> dict:
    """Gets the salient term in a sector
    Args:
        labelled: a labelled df
        glass_sector: glass company df (bring tokens back)
        top_terms: top salient terms to consider
    Returns:
        A dict mapping sectors vs tokens
    """
    labelled_ = labelled.copy()
    labelled_["token_filtered"] = labelled_["org_id"].map(
        glass_sector.set_index("org_id")["token_filtered"]
    )

    doc_term_new_sectors = make_tfidf_mat(
        make_doc_term_matrix(labelled_, "label", "token_filtered")
    )

    results = {}

    for sector in doc_term_new_sectors.index.tolist():
        salient = (
            doc_term_new_sectors.loc[sector, :]
            .sort_values(ascending=False)[:top_terms]
            .index.tolist()
        )
        results[sector] = ", ".join(salient)
    return results


def assess_sector_homogeneity(token_list: list, w2v_model) -> float:
    """Calculate median difference between tokens in a sector title
    Args:
        token_list: list of salient tokens in a sector name
        w2v_model: word2vec we use to get word similarities
    Returns:
        median of pairwise distances between terms in a description
    """

    tokens = [x for x in token_list.split(", ") if x in set(w2v_model.wv.vocab.keys())]

    combs = combinations(tokens, 2)
    results = []
    for c in combs:
        results.append(w2v_model.wv.similarity(c[0], c[1]))

    return np.median(results)


def get_all_sector_homogeneities(
    comm_terms: dict, w2v_model: gensim.models.word2vec.Word2Vec
) -> pd.DataFrame:
    """Create lookup between communities, salient terms and internal homogeneities
    Args:
        comms_term: salient terms in a community
        w2v model: vector representation of tokens in company descriptions
    Returns:
        median similarities in community names
    """

    salient_similarities = {
        s: assess_sector_homogeneity(comm_terms[s], w2v_model)
        for s in comm_terms.keys()
    }
    comm_terms_homogeneity = (
        pd.Series(comm_terms)
        .reset_index(name="salient_terms")
        .assign(median_similarity=lambda x: x["index"].map(salient_similarities))
    )
    return comm_terms_homogeneity


def make_comm_cooccurence_list(tag_counts_df: pd.DataFrame, thres: int = 2) -> dict:
    """Create dict of sector co-occurrences in company definitions
    Args:
        tag_counts_df: counts of each sector occurrence in a company
        thres: presence threshold to consider a sector related to a company
    Returns:
        A dict looking up company ids and sectors that occur in them

    """

    res = {}

    for _id, vals in tag_counts_df.iterrows():
        res[_id] = [x for x in vals.loc[vals > thres].index]
    return res
