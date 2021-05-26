import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

import industrial_taxonomy
from industrial_taxonomy.scripts.tag_companies import save_model, save_lookup
from industrial_taxonomy.scripts.extract_communities import get_glass_tokenised
from industrial_taxonomy.taxonomy.taxonomy_filtering import (
    make_doc_term_matrix,
    make_tfidf_mat,
    filter_salient_terms,
)
from industrial_taxonomy.taxonomy.topic_modelling import (
    train_model,
    post_process_model_clusters,
)

project_dir = industrial_taxonomy.project_dir
data_path = f"{project_dir}/data/processed"


def get_top_salient_terms(tfidf: pd.DataFrame, top: int = 10) -> dict:
    """Gets top tfidf terms in a sector"""
    salient_terms = {}

    for sector, r in tfidf.iterrows():

        sort = r.sort_values(ascending=False).index.tolist()[:top]

        salient_terms[sector] = set(sort)

    return salient_terms


def calculate_cluster_homogeneity(topic_mix: pd.DataFrame) -> float:
    """Calculates mean distance from the centre of a cluster"""

    topic_arr = np.array(topic_mix)

    centroid = topic_arr.mean(axis=0)

    dists = []

    for company in topic_arr:

        dists.append(cosine(company, centroid))

    return np.mean(dists)


def make_company_descr_id(gt: pd.DataFrame, sic: str):
    """Extract company tokenised descriptions and ids in a sic codd"""

    companies = (
        gt.loc[[len(x) > 0 for x in gt["tokens_clean"]]]
        .query(f"sic4=='{sic}'")
        .reset_index(drop=True)[["org_id", "tokens_clean"]]
    )

    ids, text = [
        [r[var] for _, r in companies.iterrows()] for var in ["org_id", "tokens_clean"]
    ]
    return ids, text


def make_cluster_names(
    text: list, ids: list, cluster_assignment: dict, top: int = 30, thr: float = 0.1
) -> dict:
    """Return a lookup between clusters and their names"""

    sector_tok = pd.DataFrame(
        [[i, t] for i, t in zip(ids, text)], columns=["org_id", "tokenised"]
    ).assign(sector=lambda df: df["org_id"].map(cluster_assignment))

    doc_term_mat = make_doc_term_matrix(
        sector_tok, sector="sector", tokens="tokenised", min_occurrence=5
    )

    tf_idf = make_tfidf_mat(doc_term_mat)
    cluster_salient = get_top_salient_terms(tf_idf, top=top)

    salient_frequent = filter_salient_terms(cluster_salient, thres=thr)

    salient_final = {
        k: "_".join([el for el in v if el not in salient_frequent])
        for k, v in cluster_salient.items()
    }
    return salient_final


def make_cluster_homogeneities(
    topic_mix: pd.DataFrame, cluster_assignment: dict, cluster_names: dict
) -> pd.DataFrame:
    """Calculate mean distance from centroid in a cluster"""
    cluster_sets = {
        sector: [k for k, v in cluster_assignment.items() if v == sector]
        for sector in set(cluster_assignment.values())
    }
    topic_mix.index = [int(x) for x in topic_mix.index]

    cluster_homogeneities = {
        clust: calculate_cluster_homogeneity(
            topic_mix.loc[topic_mix.index.isin(cluster_sets[clust])]
        )
        for clust in set(cluster_sets.keys())
    }

    cluster_hom_df = pd.DataFrame(
        {"cluster": cluster_names.keys(), "name": cluster_names.values()}
    ).assign(hom=lambda df: df["cluster"].map(cluster_homogeneities))

    return cluster_hom_df


def topic_model_sic(gt, s):
    """"""
    ids, text = make_company_descr_id(gt, s)

    logging.info("training model")
    model = train_model(text, ids)

    topic_mix, cluster_assignment = post_process_model_clusters(
        model, top_level=0, cl_level=0, n="all", prefix=f"{s}_"
    )

    cluster_names = make_cluster_names(text, ids, cluster_assignment)

    cluster_homogeneities = make_cluster_homogeneities(
        topic_mix, cluster_assignment, cluster_names
    )
    return model, cluster_assignment, cluster_names, cluster_homogeneities


def merge_dicts(comm_container: list) -> dict:
    """Flattens a list of dictionaries into a dictionary"""
    merged_communities = {}
    for _dict in comm_container:
        for k, v in _dict.items():
            merged_communities[k] = v
    return merged_communities


if __name__ == "__main__":
    logging.info("Reading data")
    gt = get_glass_tokenised()

    sector_counts = gt["sic4"].value_counts()
    selected_sectors = sector_counts.loc[sector_counts > 2000].index.tolist()

    logging.info(len(selected_sectors))

    topsbm_models = []
    model_mdl = []
    cluster_allocations_all = []
    cluster_names_all = []
    cluster_homog_all = []

    for s in selected_sectors:
        logging.info(s)

        try:
            (
                model,
                cluster_assignment,
                cluster_names,
                cluster_homogeneities,
            ) = topic_model_sic(gt, s)

            logging.info("appending results")
            topsbm_models.append(model)
            model_mdl.append([s, model.mdl])
            cluster_allocations_all.append(cluster_assignment)
            cluster_names_all.append(cluster_names)
            cluster_homog_all.append(cluster_homogeneities)
        except Exception as e:
            logging.info(e)

    save_model(topsbm_models, "top_sbm_models")
    save_lookup(merge_dicts(cluster_allocations_all), "topsbm_cluster_allocations")
    save_lookup(merge_dicts(cluster_names_all), "topsbm_cluster_names")

    pd.DataFrame(model_mdl, columns=["sector", "model"]).to_csv(
        f"{data_path}/topsbm_mdl.csv", index=False
    )
    pd.concat(cluster_homog_all).to_csv(f"{data_path}/topsbm_homog.csv", index=False)
