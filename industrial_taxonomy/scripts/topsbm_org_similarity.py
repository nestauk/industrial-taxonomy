import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from industrial_taxonomy.scripts.extract_communities import get_glass_tokenised
from industrial_taxonomy.taxonomy.post_processing_functions import (
    train_d2v,
    make_average_docvecs,
    save_table,
)
from industrial_taxonomy.scripts.tag_companies import get_lookup, save_model
import industrial_taxonomy

project_dir = industrial_taxonomy.project_dir


def calculate_distance(org_id: str, sector_centroids: pd.DataFrame, d2v):

    similarities = []

    centroids_array = np.array(sector_centroids)

    for sector in centroids_array:

        sim = 1 - cosine(d2v.docvecs[org_id], sector)
        similarities.append(sim)

    return pd.Series(similarities, index=sector_centroids.index, name=org_id)


if __name__ == "__main__":

    logging.info("TEST RUN")
    logging.info("Reading data")
    gt = get_glass_tokenised().loc[:3000]

    id_to_cluster_lookup = {
        int(k): v for k, v in get_lookup("topsbm_cluster_allocations").items()
    }
    gt["sector"] = gt["org_id"].map(id_to_cluster_lookup)

    logging.info("Training doc2vec")
    tagged_docs = [
        TaggedDocument(r["tokens_clean"], [r["org_id"]]) for _, r in gt.iterrows()
    ]

    d2v = train_d2v(tagged_docs, min_count=50, dimensions=100)

    logging.info("Calculating sector centroids")
    sector_sets = (
        gt.dropna(axis=0, subset=["sector"])
        .groupby("sector")["org_id"]
        .apply(lambda x: set(x))
    )

    sector_centroids = make_average_docvecs(sector_sets, d2v)

    logging.info("Calculating sector similarities")
    org_sector_similarity_df = [
        calculate_distance(_id, sector_centroids, d2v) for _id in gt["org_id"]
    ]

    logging.info("Saving outputs")
    save_model(d2v, "d2v_for_topsbm")
    save_table(sector_centroids, "topsbm_sector_centroids", index = True)
    save_table(pd.DataFrame(org_sector_similarity_df), "topsbm_org_sector_similarity", 
               index= True)
