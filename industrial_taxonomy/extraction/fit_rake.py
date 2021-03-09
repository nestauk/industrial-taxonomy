# Fit RAKE (Rapid Keyword Extracting Algorithm)
import logging
import pandas as pd
from rake_nltk import Rake, Metric
from industrial_taxonomy.extractor.utils import get_top_kws


def get_rake_phrases(
    company_descriptions,
    metric=Metric.DEGREE_TO_FREQUENCY_RATIO,
    top_t=10,
    min_thres=1,
    verbose=True,
    **kwargs,
) -> pd.Series:
    """Gets RAKE phrases from company descriptions
    Args:
        company_descriptions (list): list of company descriptions
        metric (Metric object): Metric to score keywords
        top_t (int): Number of terms to return
        min_thres (int): Minimum occurrence threshold for returning a keyword
        verbose (boolean): if we want to get progress updates
    Returns a pd.Series with keywords and their number of occurrences
    """

    keywords = []
    for en, c in enumerate(company_descriptions):
        # Give an update if needed
        if verbose is True:
            if en % 1000 == 0:
                logging.info(f"processed {en} documents")
        # Initialise RAKE and extract keywords / keyphrases
        r = Rake(ranking_metric=metric, **kwargs)
        r.extract_keywords_from_text(c)
        keywords.append(r.ranked_phrases[:top_t])

    top_kws = get_top_kws(keywords, min_thres)

    return top_kws
