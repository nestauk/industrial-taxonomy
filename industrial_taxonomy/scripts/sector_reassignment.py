# Reassign companies to their closest sector in embedding space

import pickle
import logging
import faiss
import pandas as pd
import numpy as np
import industrial_taxonomy
from industrial_taxonomy.getters.processing import (
    get_firm_embeddings,
    get_company_sector_lookup,
    get_sector_name_lookup,
)

project_dir = industrial_taxonomy.project_dir


def calculate_centroids(emb, org_sector_lookup):
    """Calculates embedding sector centroids"""

    emb_ = emb.assign(sector=lambda df: df.index.map(org_sector_lookup)).dropna(
        axis=0, subset=["sector"]
    )

    sect_centroids = (
        emb_.reset_index(drop=False)
        .melt(id_vars=["org_id", "sector"], var_name="dimension")
        .groupby(["sector", "dimension"])["value"]
        .mean()
        .reset_index(name="mean")
        .pivot_table(index="sector", columns="dimension", values="mean")
    )

    sector_names = {n: name for n, name in enumerate(sect_centroids.index)}
    emb_ids = {n: ind for n, ind in enumerate(emb_.index)}

    sect_centroids_arr = np.array(sect_centroids).astype("float32")
    emb_all_arr = np.ascontiguousarray(
        np.array(emb_.drop(axis=1, labels=["sector"])).astype("float32")
    )

    return emb_, sect_centroids_arr, sector_names, emb_all_arr, emb_ids


def get_closest_neighbours(sect_centroids_arr, emb_all_arr, k=5):
    """Gets closest sectors to all organisations"""

    d = np.size(sect_centroids_arr, 1)
    index = faiss.IndexFlat(d, faiss.METRIC_L1)
    index.add(sect_centroids_arr)
    D, I = index.search(emb_all_arr, k)
    return D, I


def sector_reassignment(emb, org_sector_lookup, k=5):
    """Reassigns sectors to their closest neighbour"""

    logging.info("Calculating centroids")

    emb_, sect_centroids_arr, sector_names, emb_all_arr, emb_ids = calculate_centroids(
        emb, org_sector_lookup
    )

    logging.info("Finding closest neighbors")
    D, I = get_closest_neighbours(sect_centroids_arr, emb_all_arr, k=k)

    new_org_sector_lu = {
        emb_ids[n]: sector_names[reas] for n, reas in enumerate(I[:, 0])
    }
    transitions = {k: [org_sector_lookup[k], v] for k, v in new_org_sector_lu.items()}
    distances = D[:, 0]

    return emb_, new_org_sector_lu, transitions, distances


def sector_reassignment_iterative(emb, company_label_lookup, iter_n=10):
    """Iterative applications of sector reassignment"""

    n = 0
    org_sector_container = [company_label_lookup]
    transition_container = []
    distance_container = []

    emb_ = emb.copy()

    while n < iter_n:
        logging.info(f"iteration {n}")
        results = sector_reassignment(emb_, org_sector_container[-1])
        org_sector_container.append(results[1])
        transition_container.append(results[2])
        distance_container.append(results[3])

        n += 1

    return org_sector_container, transition_container, distance_container


def get_closest(emb, org_sector_lookup, k=10):
    logging.info("Calculating centroids")
    emb_, sect_centroids_arr, sector_names, emb_all_arr, emb_ids = calculate_centroids(
        emb, org_sector_lookup
    )

    logging.info("Getting closest neighbours")
    D, I = get_closest_neighbours(sect_centroids_arr, emb_all_arr, k=k)

    I_named = [[sector_names[s] for s in close] for close in I]

    return I_named, D


def save_reassignment_outputs(
    org_sector_container,
    transition_container,
    distance_container,
    closest_sectors,
    distances,
):

    res = {}
    res["org_sector_container"] = org_sector_container
    res["transition_container"] = transition_container
    res["distance_container"] = distance_container
    res["sector_cooccurrences"] = closest_sectors
    res["sector_distances"] = distances

    with open(f"{project_dir}/data/processed/reassignment_analysis.p", "wb") as outfile:
        pickle.dump(res, outfile)


if __name__ == "__main__":

    logging.info("Getting data")
    emb = get_firm_embeddings().set_index("org_id")
    emb.columns = [int(x) for x in emb.columns]

    company_label_lookup = get_company_sector_lookup()
    label_name_lookup = get_sector_name_lookup()

    (
        org_sector_container,
        transition_container,
        distance_container,
    ) = sector_reassignment_iterative(emb, company_label_lookup, iter_n=15)

    closest_sectors, distances = get_closest(emb, org_sector_container[-1], k=10)

    save_reassignment_outputs(
        org_sector_container,
        transition_container,
        distance_container,
        closest_sectors,
        distances,
    )
