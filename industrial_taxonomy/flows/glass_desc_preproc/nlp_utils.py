"""Generic NLP utils."""
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Set

import toolz.curried as t
from gensim import models
from gensim.corpora import Dictionary
from spacy.lang.en import STOP_WORDS

# STOP_WORDS = set(
#     stopwords.words("english") + list(string.punctuation) + ["\\n"] + ["quot"]
# )

SECOND_ORDER_STOP_WORDS: Set[str] = t.pipe(
    STOP_WORDS,
    lambda stops: product(stops, stops),
    t.map(lambda x: f"{x[0]}_{x[1]}"),
    set,
    lambda stops: stops | STOP_WORDS,
)


def make_ngrams(
    documents: List[List[str]], n: int = 2, phrase_kws: Optional[Dict[str, Any]] = None
) -> List[List[str]]:
    """Create ngrams using Gensim's phrases.

    Args:
        documents: Tokenized documents.
        n: The `n` in n-gram.
        phrase_kws: Passed to `gensim.models.Phrases`.

    Return:
        N-grams

    #UTILS
    """
    assert isinstance(n, int)
    if n < 2:
        raise ValueError("Pass n >= 2 to generate n-grams")

    def_phrase_kws = {
        "scoring": "npmi",
        "threshold": 0.25,
        "min_count": 2,
        "delimiter": b"_",
    }
    phrase_kws = t.merge(def_phrase_kws, phrase_kws or {})

    step = 1
    while step < n:
        phrases = models.Phrases(documents, **phrase_kws)
        bigram = models.phrases.Phraser(phrases)
        del phrases
        tokenised = bigram[documents]
        step += 1

    return list(tokenised)


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
