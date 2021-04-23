"""Generic NLP utils."""
import string
import re
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional

import nltk
import toolz.curried as t
from gensim import models
from nltk.corpus import stopwords


nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(
    stopwords.words("english") + list(string.punctuation) + ["\\n"] + ["quot"]
)
REGEX_TOKEN_TYPES = {
    "URL": r"(http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),](?:%[0-9a-f][0-9a-f]))+)",
    "@word": r"(@[\w_]+)",
    "XML": r"(<[^>]+>)",
    "strip_apostrophe": r"(\w+)'\w",
    "word": r"([\w_]+)",
    "non-whitespace-char": r"(?:\S)",  # TODO: is this useful?
    "hyphenated": r"(\w+-[\w\d]+)",
}


def tokenize(text: str, tokens_re: re.Pattern) -> Iterable[str]:
    """Preprocess a raw string/sentence of text. """
    return t.pipe(
        text,
        tokens_re.findall,
        lambda tokens: chain.from_iterable(tokens) if tokens_re.groups > 1 else tokens,
        t.filter(None),
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
