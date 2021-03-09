# Fit KeyBERT
import logging
import pandas as pd
from itertools import chain
from keybert import KeyBERT
from industrial_taxonomy.extractor.utils import get_top_kws


def get_keybert_phrases(
    company_descriptions,
    top_t=10,
    min_thres=2,
    model_name="distilbert-base-nli-mean-tokens",
    batch=50,
    verbose=True,
    **kwargs,
) -> pd.Series:
    """Fit keybert to extract keywords from business descriptions
    Args:
        company_descriptions (list): list of company descriptions
        top_t (int): Number of terms to return
        batch (int): How many documents to consider in a single batch
        min_thres (int): Minimum occurrence threshold for returning a keyword
        model (keyBERT object): transformer to use
        verbose (boolean): if we want to get progress updates
    Returns a pd.Series with keywords and their number of occurrences
    """

    keywords = []
    model = KeyBERT(model_name)

    counter = 0
    while counter < len(company_descriptions):
        if verbose is True:
            if counter % 100 == 0:
                logging.info(f"processed {counter} documents")

        kb_ks = model.extract_keywords(
            company_descriptions[counter : counter + batch], top_n=top_t, **kwargs
        )

        # We flatten the keyword list
        keyword_flat = list(chain(*kb_ks))

        # Only append the keyword, not its similarity to the document
        keywords.append([x[0] for x in keyword_flat])
        counter += batch

    top_kws = get_top_kws(keywords, min_thres)

    return top_kws
