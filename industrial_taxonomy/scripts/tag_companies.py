import os
import json
import pandas as pd
import logging
import pickle
from gensim.models import Word2Vec
from industrial_taxonomy.getters.glass import get_organisation_description
from industrial_taxonomy.scripts.extract_communities import get_glass_tokenised
from industrial_taxonomy.taxonomy.company_labelling import (
    get_communities,
    tag_companies,
    make_labelled_dataset,
    get_salient_terms_community,
    get_all_sector_homogeneities,
    make_comm_cooccurence_list,
)

import industrial_taxonomy
from industrial_taxonomy import config

params = config["company_tagging"]

project_dir = industrial_taxonomy.project_dir
MODEL_PATH = f"{project_dir}/models/w2v_v2.p"
HOMOG_PATH = f"{project_dir}/data/processed/title_homog.csv"
LABELLED_PATH = f"{project_dir}/data/processed/glass_comm_labelled.csv"


def get_w2v():
    with open(MODEL_PATH, "rb") as infile:
        model = pickle.load(infile)
    return model


def save_model(mod, name):
    with open(project_dir + f"/model/{name}.p", "wb") as outfile:
        pickle.dump(mod, outfile)


def load_model(name):
    with open(project_dir + f"/model/{name}.p", "rb") as infile:
        model = pickle.load(infile)
    return model


def save_lookup(obj, name):
    with open(f"{project_dir}/data/processed/{name}.json", "w") as outfile:
        json.dump(obj, outfile)


def get_glass_labelled():
    return pd.read_csv(LABELLED_PATH)


def get_name_homogeneities():
    return pd.read_csv(HOMOG_PATH)


def get_lookup(name):
    with open(f"{project_dir}/data/processed/{name}.json", "r") as infile:
        lookup = json.load(infile)
        return lookup


if __name__ == "__main__":

    logging.info("Reading data")
    description = (
        get_organisation_description().set_index("org_id")["description"].to_dict()
    )
    glass = get_glass_tokenised()
    communities = get_communities()

    if os.path.exists(MODEL_PATH) is True:
        w2v = get_w2v()
    else:
        logging.info("Making model")
        w2v = Word2Vec(glass["token_filtered"])
        with open(MODEL_PATH, "wb") as outfile:
            pickle.dump(w2v, outfile)

    logging.info("Labelling companies")
    glass_tagged = make_labelled_dataset(
        tag_companies(glass, description, communities),
        occ_threshold=params["occ_thres_label"],
    )
    logging.info(glass_tagged.head())
    glass_tagged.to_csv(LABELLED_PATH, index=False)

    logging.info("Making community names")
    salient_terms = get_salient_terms_community(glass_tagged, glass, top_terms=10)
    homog_table = get_all_sector_homogeneities(salient_terms, w2v)

    save_lookup(salient_terms, "comm_terms_lookup")
    logging.info(homog_table.head())
    homog_table.to_csv(HOMOG_PATH, index=False)

    logging.info("Multi-tags")
    tag_counts = tag_companies(glass, description, communities, exclusive=False)
    co_occurrence_dict = make_comm_cooccurence_list(
        tag_counts, thres=params["occ_thres_tag"]
    )
    save_lookup(co_occurrence_dict, "co_occurrence")
