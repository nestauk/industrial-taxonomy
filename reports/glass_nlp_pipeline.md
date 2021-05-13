# Pre-processing of Glass business descriptions {#sec:glass_preproc}

The analysis of [@sec:topsbm] and [@sec:taxonomy] requires processing the raw descriptions extracted from business' websites by Glass into a form we can use to, for example, generate a network of words where the strength of connections between words is based on how frequently they co-occur in documents.

We build a NLP pipeline using Spacy [@spacy] and Gensim [@gensim] which converts the raw 'string' of a business website description into an ordered list of IMPORTANT(?) 'tokens' where each token is a unigram, bigram or trigram composed of the lemmatised form of a word or an (uppercased) entity type label (for a subset of entity types).

For example: `"I went to the Eiffel tower and visited my friend Jacques"` -> `["went", "GPE", "visit", "friend", "PERSON"
`]

## Pipeline steps

Steps 1-8 are performed or rely on information extracted using the Spacy `en_core_web_lg` model, whilst steps 9-10 are performed or rely on information extracted using Gensim.

1. Tokenisation
4. Named entity recognition (NER) - Predict named entities using a transition-based method
    TODO: summarise how these methods work https://spacy.io/api/architectures#TransitionBasedParser 
5. Lemmatise - Assign base forms to tokens
6. Merge entities - Merges series of tokens predicted by Spacy to be an entity into a single token
7. Filter
   - stopwords
   - punctuation
   - whitespace
8. Extract to list of strings - lemmatise or entity form
9. Generate n-grams - Use `gensim` to generate bi-grams, requiring that a bigram occurs at least 10 times and that the normalise pointwise mutual information (NPMI)[^NPMI] is 0.25 or higher.

10. Filter
   - short tokens
   - combinations of stop words
   - Words with low and very high frequency (those occurring less than 10 times and in more than 90% of documents)


# Appendix

## Token lemmatisation/remapping {#sec:remapping}

Tokens that are entities in the following categories are renamed to correspond to their entity category (uppercased):

 - CARDINAL
 - DATE
 - GPE
 - LOC
 - MONEY
 - NORP
 - ORDINAL
 - ORG
 - PERCENT
 - PERSON
 - QUANTITY
 - TIME

I hypothesised that replacing these entities with their entity type name helps keep more information in the bag of words representation, particularly when entity types can be formed into n-grams with other words. 
The alternative is that individual dates, people, organisation names etc. are too infrequent in the corpus to contribute information to the topic modelling approach.

Several entity categories such as `WORK_OF_ART`, `LANGUAGE`, `LAW`, `EVENT` were excluded as an empirical assessment of the classifications appeared in accurate.
Furthermore, `PRODUCT` was left out because in this problem context (generating an industrial taxonomy) this is valuable information that we do not wish to homogenise.

Tokens that are not entities are replaced with their lowercase lemmatised form.

## Potential Improvements

- Consider parsing documents with BeautifulSoup and converting XML to visible text before running these pipelines

