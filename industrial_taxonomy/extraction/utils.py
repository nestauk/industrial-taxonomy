# Utilities for keyword extraction
from itertools import chain
import pandas as pd


def return_top_terms(_list, min_thres):
    """Takes a list of keyword lists and returns their frequencies if
    above a threshold
    """
    top_terms = pd.Series(chain(*_list)).value_counts()
    return top_terms.loc[top_terms > min_thres]
