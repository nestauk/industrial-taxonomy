import os
import pandas as pd
import logging
from industrial_taxonomy.scripts.extract_communities import (
    get_glass_tokenised,
)
from industrial_taxonomy.scripts.tag_companies import (
    get_lookup,
    save_lookup,
    get_glass_labelled,
    get_name_homogeneities,
)
from industrial_taxonomy.taxonomy.post_processing_functions import (
    save_model,
    load_model,
    save_table,
    make_co_occurrence_table,
    make_distance_matrix,
    find_duplicated_communities,
    make_labelled_set,
    train_d2v,
    make_average_docvecs,
    find_closest_sector,
)
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import industrial_taxonomy

project_dir = industrial_taxonomy.project_dir

D2V_PATH = f"{project_dir}/models/d2v.p"

if __name__ == "__main__":
    logging.info("Reading data")
    comm_names = get_lookup("comm_terms_lookup")
    comm_names_2 = {k: k + ": " + v for k, v in comm_names.items()}
    homog = get_name_homogeneities()
    high_homog = homog.query("median_similarity>0.5")["index"].tolist()
    comm_coocc = get_lookup("co_occurrence")
    g_lab = get_glass_labelled()
    glass_tok = get_glass_tokenised()

    # Find duplicated communities and merge them
    logging.info("finding duplicated communities")
    co_occ_table = make_co_occurrence_table(comm_coocc)
    sim_df = make_distance_matrix(co_occ_table.T, similarity=True, return_df=True)
    merged_comms, merged_comms_names = find_duplicated_communities(
        sim_df[high_homog], comm_names
    )

    save_lookup(merged_comms, "merged_communities")
    save_lookup(merged_comms_names, "merged_comm_names")

    # Relabel duplicated communities
    logging.info("Making labelled dataset")
    labelled_set = make_labelled_set(g_lab, merged_comms, count_thres=10)
    labelled_set_high_homog = labelled_set.loc[
        [x in high_homog + list(merged_comms.values()) for x in labelled_set.index]
    ]

    if os.path.exists(D2V_PATH) is False:
        logging.info("Training d2v model")
        tagged_docs = [
            TaggedDocument(r["token_filtered"], [r["org_id"]])
            for _, r in glass_tok.iterrows()
        ]
        d2v = train_d2v(tagged_docs)
        save_model(d2v, "d2v")
    else:
        logging.info("Loading d2v model")
        d2v = load_model("d2v")

    labelled_average_vectors = make_average_docvecs(labelled_set_high_homog, d2v)

    # Labelled dataset with closest
    # Unlabelled dataset
    logging.info("finding closest companies")
    unl = g_lab.query("label=='unlabelled'").reset_index(drop=True)
    unl["token_filtered"] = unl["org_id"].map(
        glass_tok.set_index("org_id")["token_filtered"]
    )

    closest_sector_df = pd.DataFrame(
        [
            find_closest_sector(row, labelled_average_vectors, d2v, counter)
            for counter, row in unl.iterrows()
        ],
        columns=["sector", "similarity"],
    ).assign(org_id=unl["org_id"])

    save_table(closest_sector_df, "unlabelled_closest_sector_V2")
