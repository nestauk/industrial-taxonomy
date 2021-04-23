"""Tokenises and ngrams Glass organisation descriptions."""
import json
from typing import Any, Generator

import toolz.curried as t
from metaflow import FlowSpec, step, Parameter, IncludeFile, JSONType, conda_base

from nlp_utils import (
    make_ngrams,
    filter_frequency,
    post_token_filter,
    spacy_pipeline,
    bag_of_words as bag_of_words_,
    ENTS,
)


@t.curry
def log_(msg: str, x: Any, how=print) -> Any:
    how(msg)
    return x


@conda_base(
    libraries={
        "spacy": ">=3.0",
        "toolz": ">0.11",
        "gensim": ">=4.0",
        "spacy-model-en_core_web_lg": ">=3.0",
    },
    python="3.8",
)
class EscoeNlpFlow(FlowSpec):
    n_process = Parameter(
        "n-process",
        help="The number of processes to use with spacy (default: 1)",
        type=int,
        default=1,
    )
    n_gram = Parameter(
        "n-gram",
        help="The `N` in N-gram",
        type=int,
        default=2,
    )
    entity_mappings = Parameter(
        "entity-mappings",
        help="",
        type=JSONType,
        default=json.dumps(ENTS),
    )
    input_file = IncludeFile(
        "input-file",
        help="JSON file, mapping document id to document text."
        " Structure: Dict[str: str]",
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
        """Load data and run the NLP pipeline, returning tokenised documents."""

        data = json.loads(self.input_file)
        self.documents = list(data.values())
        self.keys = list(data.keys())
        print(f"Received {len(data)} documents")

        nlp = spacy_pipeline()
        bag_of_words = t.curry(bag_of_words_, entity_mappings=self.entity_mappings)
        tokens = t.pipe(
            self.pop_documents(),
            # Step: Remove HTML
            # TODO ?
            # Step: Spacy
            t.curry(nlp.pipe, n_process=self.n_process),
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

        self.documents = dict(zip(self.keys, tokens))
        print(f"Processed {len(self.documents)} documents")
        assert len(self.documents) == len(self.keys), (
            "Number of document ID's and processed documents does not match... "
            f"{len(self.keys)} != {len(self.documents)}"
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    EscoeNlpFlow()
else:
    # %%

    nlp = spacy_pipeline()
    with open("input_data.json") as f:
        documents = list(json.load(f).values())

    # %%

    bag_of_words_(nlp(documents[0]))
    # %%

    print("curry")
    bag_of_words = t.curry(bag_of_words_, entity_mappings=ENTS)
    bag_of_words(nlp(documents[0]))

# %%
