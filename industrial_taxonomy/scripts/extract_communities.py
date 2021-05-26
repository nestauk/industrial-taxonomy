# Extracts communities from SIC sectors

import os
import ast
import pickle
import json
import pandas as pd
import logging
from cdlib import algorithms, ensemble
from industrial_taxonomy.taxonomy.taxonomy_filtering import (
    make_glass_for_taxonomy,
    text_processing,
)
from industrial_taxonomy.getters.processing import get_config
from industrial_taxonomy.taxonomy.taxonomy_community import extract_communities
import industrial_taxonomy
from industrial_taxonomy import config

project_dir = industrial_taxonomy.project_dir

TOK_PATH = f"{project_dir}/data/processed/glass_tokenised_V2.csv"


# params = config["community_extraction"]
params = get_config("community_extraction")
k_param = ensemble.Parameter(name="k", start=20, end=40, step=10)

method_pars = {
    algorithms.louvain: [
        ensemble.Parameter(name="resolution", start=0.7, end=1, step=0.1)
    ],
    algorithms.leiden: [ensemble.BoolParameter(name="weights", value="weight")],
    # algorithms.em:[k_param],
    # algorithms.async_fluid:[k_param],
    algorithms.agdl: [
        ensemble.Parameter(name="number_communities", start=20, end=50, step=10),
        ensemble.Parameter(name="kc", start=4, end=10, step=2),
    ],
    algorithms.chinesewhispers: [],
    # algorithms.girvan_newman: [
    #     ensemble.Parameter(name="level", start=20, end=50, step=15)
    # ],
    algorithms.label_propagation: [],
    algorithms.markov_clustering: [],
    algorithms.sbm_dl_nested: [
        ensemble.Parameter(name="B_min", start=20, end=50, step=10)
    ],
}


def get_large_sectors(glass_sector: pd.DataFrame, size: float = 500) -> list:
    """Get sectors above a certain size
    Args:
        glass_sector: glass companies
        size: minimum size for inclusion
    Returns:
        set of large sectors
    """
    sic4_counts = glass_sector["sic4"].value_counts()
    my_sectors = sic4_counts.loc[sic4_counts >= size].index.tolist()
    return my_sectors


def save_extraction_outputs(results: list):
    """Saves community extraction outputs"""
    logging.info("Saving results")
    with open(f"{project_dir}/data/processed/sector_networks.p", "wb") as outfile:
        pickle.dump(results[0], outfile)
    with open(f"{project_dir}/data/processed/sector_communities.json", "w") as outfile:
        json.dump(results[1], outfile)

    results[2].to_csv(
        f"{project_dir}/data/processed/partition_similarities.csv", index=False
    )

    results[3].to_csv(
        f"{project_dir}/data/processed/partition_summaries.csv", index=False
    )


def get_glass_tokenised():
    """Reads the glass tokenised data"""

    g = pd.read_csv(
        TOK_PATH,
        dtype={"sic4": str},
        converters={
            "tokens": ast.literal_eval,
            "tokens_clean": ast.literal_eval,
            "token_filtered": ast.literal_eval,
        },
    )
    return g


def get_summaries(summary_name):
    table_loc = f"{project_dir}/data/processed/{summary_name}.csv"
    return pd.read_csv(table_loc, dtype={"sic4": str})


if __name__ == "__main__":

    if os.path.exists(TOK_PATH) is True:
        logging.info("reading preprocessed data")
        glass_processed = get_glass_tokenised()
    else:
        logging.info("Making tokenised data")
        glass = make_glass_for_taxonomy()
        glass_processed = text_processing(glass, params["salient_filter_thres"])
        logging.info(glass_processed[["tokens_clean", "token_filtered"]].head())
        glass_processed.to_csv(TOK_PATH)

    # sectors = ["6201", "9609"]

    sectors = get_large_sectors(glass_processed, size=params["min_sector_size"])
    extraction_outputs = extract_communities(glass_processed, sectors, method_pars)

    save_extraction_outputs(extraction_outputs)
