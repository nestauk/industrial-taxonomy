"""Functions to filter tokens the SIC taxonomy
"""
import re
import logging
import string
import pandas as pd
from itertools import chain
from industrial_taxonomy.getters.glass import (
    get_description_tokens,
    get_organisation,
    get_organisation_description,
)
from industrial_taxonomy.getters.glass_house import get_glass_house
from industrial_taxonomy.getters.companies_house import get_sector
from industrial_taxonomy.taxonomy import make_network_from_coocc, label_comms
from industrial_taxonomy.utils.sic_utils import extract_sic_code_description
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

NERS = [
    "CARDINAL",
    "DATE",
    "GPE",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "QUANTITY",
    "TIME",
]


def term_freqs(list_of_lists: list) -> pd.Series:
    """Get term frequencies in a corpus
    Args:
        list_of_lists: list of documents where every document is a list of tokens
    Returns:
        token frequencies in the corpus
    """

    counts = pd.Series(chain(*list_of_lists)).value_counts()
    return counts


def drop_ners(tokenised: list, ners: set = NERS) -> list:
    """Remove named entity recognitions from a document
    Args:
        tokenised: document as list of tokens
        ners: list of spacy named entity identifiers
    Returns:
        document without the ners
    """

    return [x.lower() for x in tokenised if x not in ners]


def make_glass_for_taxonomy() -> pd.DataFrame:
    """Make dataset to create the taxonomy
    Returns:
        table with glass sectors and tokenised descriptions
    """
    logging.info("Making glass dataset for processing")

    glass_w_tokens = get_organisation().merge(
        pd.Series(get_description_tokens(), name="tokens"),
        left_on="org_id",
        right_index=True,
    )

    gh = get_glass_house()

    sect = get_sector()

    descr = get_organisation_description

    glass_sector = (
        gh.query("score>=75")
        .merge(sect, on="company_number")
        .sort_values(
            "score", ascending=False
        )  # This way we can keep the first duplicated (highest score)
        .drop_duplicates(subset=["company_number"], keep="first")
        .assign(sic4=lambda df: df["SIC5_code"].apply(lambda x: x[:-1]))
        .merge(glass_w_tokens, on="org_id")
    )
    return glass_sector


def make_doc_term_matrix(
    glass_sector: pd.DataFrame, sector: str = "sic4", tokens: str = "tokens_clean"
) -> pd.DataFrame:
    """Create document - term matrix
    Args:
        glass_sector: table with glass tokenised descriptions and sectors
        sector: sector variable
        tokens: token variable
    Returns:
        document term matrix with token counts per sic code
    """

    doc_term_mat = (
        glass_sector.groupby(sector)[tokens]
        .apply(lambda x: pd.Series(chain(*list(x))).value_counts())
        .reset_index(name="count")  # Here we reshape into a doc term matrix
        .pivot_table(index=sector, columns="level_1", values="count")
        .fillna(0)
    )

    # Only keep terms that begin with a letter
    doc_term_mat = doc_term_mat[
        [x for x in doc_term_mat.columns if x[0] in string.ascii_letters]
    ]

    return doc_term_mat


def make_tfidf_mat(doc_term_mat: pd.DataFrame) -> pd.DataFrame:
    """Create tf-idf matrix
    Args:
        doc_term_mat: document term matrix
    Returns:
        tf-idf matrix
    """

    tf = TfidfTransformer()

    tfidf_mat = pd.DataFrame(
        tf.fit_transform(doc_term_mat).todense(),
        columns=doc_term_mat.columns,
        index=doc_term_mat.index,
    )
    return tfidf_mat


def extract_salient_terms(tfidf_mat: pd.DataFrame, q: float = 0.95) -> dict:
    """Extract most salient terms for each row in a tfidf matrix
    Args:
        tfidf_mat: tf idf matrix
        q: tfidf quantile over which we extract terms
    Returns:
        dictionary where keys are row indices and values a list of salient terms
    """

    salient_terms = {}

    for sic, r in tfidf_mat.iterrows():

        top_q = r.quantile(q=q)
        sort = r.sort_values(ascending=False)

        my_terms = sort.loc[sort > top_q].index.tolist()

        salient_terms[sic] = set(my_terms)

    return salient_terms


def filter_salient_terms(salient_terms: dict, thres: float = 0.5) -> list:
    """Identify salient terms that happen in many sectors
    Args:
        salient_terms: dict with salient terms per sector
        thres: max share of sectors where a salient term is allowed to appear
    Returns:
        list of salient terms to remove
    """

    all_salient_terms = set(chain(*salient_terms.values()))
    salient_sets = list(salient_terms.values())

    salient_occurrences = pd.Series(
        {
            word: sum([word in _set for _set in salient_sets])
            for word in all_salient_terms
        }
    ) / len(salient_sets)

    not_very_salient = salient_occurrences.loc[
        salient_occurrences > thres
    ].index.tolist()
    return not_very_salient


def get_promo_terms(doc_term_mat: pd.DataFrame, thres: float = 0.5) -> set:
    """Identify promotional terms
    Args:
        doc_term_mat: document term matrix
        thres: polarity threshold to identify marketing buzz
    Returns:
        set of promotional terms
    """

    sia = SentimentIntensityAnalyzer()

    polarity_scores = {
        x: sia.polarity_scores(re.sub("_", " ", x))["compound"]
        for x in doc_term_mat.columns
    }
    senti = pd.Series(polarity_scores)
    marketing_buzz = set(senti.loc[senti > thres].index)
    return marketing_buzz


def text_processing(
    glass_sector: pd.DataFrame,
    salient_q: float = 0.95,
    salient_filter_thres: float = 0.3,
    polarity_thres: float = 0.5,
) -> pd.DataFrame:
    """Preprocesses the glass data"""

    logging.info("Processing glass tokens")

    glass_sector["tokens_clean"] = glass_sector["tokens"].apply(drop_ners)
    doc_term_mat = make_doc_term_matrix(glass_sector)
    promo_terms = get_promo_terms(doc_term_mat)
    doc_term_mat = doc_term_mat[
        [x for x in doc_term_mat.columns if x not in promo_terms]
    ]
    tfidf_mat = make_tfidf_mat(doc_term_mat)
    salient_terms_sector = extract_salient_terms(tfidf_mat, salient_q)
    terms_to_drop = filter_salient_terms(salient_terms_sector, salient_filter_thres)
    salient_terms_filtered = {
        k: v - set(terms_to_drop) for k, v in salient_terms_sector.items()
    }

    glass_sector["token_filtered"] = [
        [x for x in row["tokens_clean"] if x in salient_terms_filtered[row["sic4"]]]
        for _id, row in glass_sector.iterrows()
    ]
    return glass_sector

gs = make_glass_for_taxonomy()
f = text_processing(gs)
logging.info(f['token_filtered'].head())

