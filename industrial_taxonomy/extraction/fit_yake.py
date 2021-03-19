# Fit YAKE (yet another Keyword extractor)

import logging
import pandas as pd
from yake import KeywordExtractor
from industrial_taxonomy.extraction.utils import get_top_terms


def get_yake_phrases(
    company_descriptions, top_t=10, min_thres=1, verbose=True, **kwargs
) -> pd.Series:
    """Extracts YAKE phrases from company descriptions
    Args:
        company_descriptions (list): list of company descriptions
        top_t (int): Number of terms to return
        min_thres (int): Minimum occurrence threshold for returning a keyword
        verbose (boolean): if we want to get progress updates
    Returns a pd.Series with keywords and their number of occurrences
    """
    keywords = []
    for en, c in enumerate(company_descriptions):
        # If we want process updates
        if verbose is True:
            if en % 1000 == 0:
                logging.info(f"processed {en} documents")

        # Initialise YAKE and extract keywords
        yake_extractor = KeywordExtractor(**kwargs)
        yake_ks = yake_extractor.extract_keywords(c)
        keywords.append([x[0] for x in yake_ks[:top_t]])

    top_kws = get_top_terms(keywords, min_thres)

    return top_kws
