"""Tokenises and ngrams Glass organisation descriptions."""
import re
import string
from itertools import product
from typing import Any, Dict, Generator, Iterable, List, Optional, Set

import pandas as pd
import spacy
import toolz.curried as t
from gensim.corpora import Dictionary
from metaflow import FlowSpec, step, Parameter
from nltk.stem import WordNetLemmatizer
from spacy.tokens import Doc, Token
from spacy.language import Language

import industrial_taxonomy
from industrial_taxonomy.getters.glass import get_organisation_description
from industrial_taxonomy.getters.glass_house import get_glass_house
from industrial_taxonomy.nlp import make_ngrams, tokenize, STOP_WORDS, REGEX_TOKEN_TYPES

N_TEST_DOCS = 1_000
SECOND_ORDER_STOP_WORDS: Set[str] = t.pipe(
    STOP_WORDS,
    lambda stops: product(stops, stops),
    t.map(lambda x: f"{x[0]}_{x[1]}"),
    set,
    lambda stops: stops | STOP_WORDS,
)


@t.curry
def log_(msg: str, x: Any, how=print) -> Any:
    how(msg)
    return x


def pre_token_filter(tokens: Iterable[str]) -> Iterable[str]:
    """Pre n-gram token filter."""

    filter_re = "|".join(
        [REGEX_TOKEN_TYPES["URL"], REGEX_TOKEN_TYPES["@word"], REGEX_TOKEN_TYPES["XML"]]
    )

    def predicate(token: str) -> bool:
        return (
            # At least one ascii letter
            any(x in token for x in string.ascii_lowercase)
            # No URL's | twitter ID's
            and not re.match(filter_re, token)
        )

    return filter(predicate, tokens)


def post_token_filter(tokens: Iterable[str]) -> Iterable[str]:
    """Post n-gram token filter."""

    def predicate(token: str) -> bool:
        return (
            # No short words
            (not len(token) <= 2)
            # No stopwords
            and (token not in SECOND_ORDER_STOP_WORDS)
        )

    return filter(predicate, tokens)


def token_converter(tokens: Iterable[str]) -> Iterable[str]:
    """Convert tokens."""

    def convert(token: str) -> str:
        return token.lower().replace("-", "_")

    return map(convert, tokens)


@t.curry
def filter_frequency(
    documents: List[str], kwargs: Optional[Dict[str, Any]] = None
) -> Iterable[str]:
    """Filter `documents` based on token frequency corpus."""
    dct = Dictionary(documents)

    default_kwargs = dict(no_below=10, no_above=0.9, keep_n=1_000_000)
    if kwargs is None:
        kwargs = default_kwargs
    else:
        kwargs = t.merge(default_kwargs, kwargs)

    dct.filter_extremes(**kwargs)
    return t.pipe(
        documents,
        t.map(lambda document: [token for token in document if token in dct.token2id]),
    )


class GlassDescPreprocFlow(FlowSpec):
    n_gram = Parameter(
        "n-gram",
        help="The `N` in N-gram",
        type=int,
        default=2,
    )
    test_mode = Parameter(
        "test_mode",
        help="Whether to run in test mode (fetch a subset of data)",
        type=bool,
        default=True,
    )
    n_docs = Parameter(
        "n_docs",
        help="Number of documents to process",
        type=int,
        default=-1,
    )

    def pop_documents(self) -> Generator[str, None, None]:
        """Destructively yield from `self.documents`."""
        self.documents.reverse()
        n = len(self.documents)
        i = 0
        while i < n:
            yield self.documents.pop()
            i += 1

    @step
    def start(self):
        match_threshold = industrial_taxonomy.config["params"]["match_threshold"]

        def sample(df: pd.DataFrame, n_docs: int) -> pd.DataFrame:
            """Sample dataframe"""
            if n_docs == -1:
                return df
            else:
                return df.sample(n=self.n_docs, random_state=889)

        documents = (
            get_organisation_description()
            # Only orgs with a companies house match
            .merge(
                get_glass_house()
                .query(f"score > {match_threshold}")
                .drop(["company_number"], axis=1),
                on="org_id",
            )
            .sort_values("score", ascending=False)
            .drop(["score"], axis=1)
            .pipe(sample, self.n_docs)
            .head(N_TEST_DOCS if self.test_mode else None)[["org_id", "description"]]
        )

        print(f"{documents.shape[0]} documents")

        self.documents = documents.description.tolist()
        self.keys = documents.org_id.tolist()

        self.next(self.pipeline1, self.pipeline2)

    @step
    def pipeline1(self):
        lemmatiser = WordNetLemmatizer()

        tokens_re = re.compile(
            "|".join(
                [
                    REGEX_TOKEN_TYPES["URL"],
                    REGEX_TOKEN_TYPES["@word"],
                    REGEX_TOKEN_TYPES["XML"],
                    REGEX_TOKEN_TYPES["strip_apostrophe"],
                    REGEX_TOKEN_TYPES["hyphenated"],
                    REGEX_TOKEN_TYPES["word"],
                ]
            ),
            re.VERBOSE | re.IGNORECASE,
        )

        tokenize_ = t.curry(tokenize, tokens_re=tokens_re)

        tokens = t.pipe(
            self.pop_documents(),
            # Tokenise and filter
            t.map(t.compose(list, pre_token_filter, token_converter, tokenize_)),
            list,
            log_("Tokenised"),
            # Filter low frequency terms (want to keep high frequency terms)
            filter_frequency(kwargs={"no_above": 1}),
            list,
            log_("Frequency filter 1"),
            # N-gram
            t.curry(make_ngrams, n=self.n_gram),
            # Lemmatise
            t.map(lambda document: [lemmatiser.lemmatize(word) for word in document]),
            # Filter ngrams: combination of stopwords, e.g. `of_the`
            t.map(t.compose(list, post_token_filter)),
            list,
            log_("N-grammed, lemmatised, and post filtered"),
            # Filter ngrams:  low (and very high) frequency terms
            filter_frequency,
            list,
            log_("Frequency filter 1"),
        )

        self.docs = dict(zip(self.keys, tokens))
        self.next(self.join)

    @step
    def pipeline2(self):

        nlp = spacy_pipeline()

        tokens = t.pipe(
            self.pop_documents(),
            # Step: Remove HTML
            # Step: Spacy
            t.curry(nlp.pipe, n_process=2),
            list,
            log_("Spacified"),
            # Step: bag of words
            t.map(bag_of_words),
            list,
            log_("Converted to Bag of Words"),
            # # Filter low frequency terms (want to keep high frequency terms)
            filter_frequency(kwargs={"no_above": 1}),
            list,
            # N-gram
            t.curry(make_ngrams, n=self.n_gram),
            # Filter ngrams: combination of stopwords, e.g. `of_the`
            t.map(t.compose(list, post_token_filter)),
            list,
            # Filter ngrams:  low (and very high) frequency terms
            filter_frequency,
            list,
        )

        self.docs = dict(zip(self.keys, tokens))
        self.next(self.join)

    @step
    def join(self, inputs):
        self.docs_v1 = inputs.pipeline1.docs
        self.docs_v2 = inputs.pipeline2.docs
        self.next(self.end)

    @step
    def end(self):
        pass


def bag_of_words(doc: Doc) -> List[str]:
    """Convert spacy document to bag of words for topic modelling."""

    def extract_text(token: Token) -> str:
        """Extract text from a spacy token."""
        if token.ent_type_ in {
            "CARDINAL",
            "DATE",
            # "EVENT",
            # "FAC",
            "GPE",
            # "LANGUAGE",
            # "LAW",
            "LOC",
            "MONEY",
            "NORP",  # ?
            "ORDINAL",
            "ORG",
            "PERCENT",
            "PERSON",
            # "PRODUCT",  # Loses too much info, we're interested in industry
            "QUANTITY",
            "TIME",
            # "WORK_OF_ART",
        }:
            return token.ent_type_
        else:
            return token.lemma_

    return t.pipe(
        doc,
        t.filter(lambda x: not (x.is_stop or x.is_punct or x.is_space)),
        t.map(extract_text),
        list,
    )


def spacy_pipeline() -> Language:
    """Spacy NLP pipeline."""
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("merge_entities")
    return nlp


if __name__ == "__main__":
    GlassDescPreprocFlow()
