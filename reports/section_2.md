
# Data collection and processing

## Glass.ai

The core dataset for our analysis has been obtained from Glass, a startup that uses machine learning to collect and analyse business website data at scale.
More specifically, Glass begin from the universe of UK websites in web domain registers, identifies those that are highly likely to belong to a business, and extracts relevant information about them including their description, postcodes and sector based on an industrial taxonomy developed by LinkedIn.
In this project, we work with information about 1.8 million business websites (which according to Glass account for 90% of UK business websites) collected in May and June 2020.

The granular business descriptions contained within this dataset can be used to understand a business' economic activities at a higher level of resolution than is possible using the SIC taxonomy.

## Labelling business websites with SIC codes by matching to Companies House {#sec:matching}

In order to obtain SIC codes for business websites we match businesses in the Glass dataset to the Companies house business registry.
We use the monthly data snapshots for May, June, and July 2020 - available at the time from the Companies house website - as this corresponds to the period of time for which we have Glass data.

<!-- BONUS:, and to a lesser extent to obtain a secondary source of address data,-->

The matching methodology matches the names of companies in Companies House with the names extracted by Glass from business websites based on their similarity according to some measure[^similarity].
Naively comparing the similarity of all combinations (~4 million Companies House companies x ~1.5 million Glass websites) of names is computationally infeasible - this would take roughly 20 years (on a single CPU) assuming we could do 10,000 similarity computations per second.
This leaves us with three options, with the third being the only one that is reasonable and possible:

1. Performing the computations on a supercomputer
2. Make a breakthrough in the field of computing by improving the performance of a fundamental algorithm by several orders of magnitudes
3. Reduce the number of pairwise comparisons by only comparing pairs that are 'likely' to be matches

[^similarity]: For example the Levenshtein distance [@levenshtein].

It may sound paradoxical that we could identify pairs that are likely to be matches without calculating their similarites up-front.
We achieve this by using two approaches:
probabilistic data structures -
namely the Minhash combined with Locality Sensitive Hashing (LSH) -
and the cosine similarity computed using a chunked dot-product [^cheating].

[^cheating]: This approach does in fact compute similarities up-front; however the similarity measure used is cheap to compute.

### Probabilistic data structures (PDS) approach

Convert company names to a set of 'k-shingles' - a sliding window $k$ characters.
For example,
for $k=4$ "acme co" would become the set {"acme", "cme ", "me c", "e co"}.

One can measure the similarity between the k-shingle sets of two company names with the Jaccard similarity - $$J(A,B) = \frac{\|A \cap B \|}{\| A \cup B\|}$$.

The Jaccard similarity can be approximated using Minhash and LSH -
a method originally used by search engines to detect near-duplicate web pages to improve their search results [@LSH].
Below we outline the high-level concepts behind the approach.
The interested reader should refer to Chapter 3 of [@mmds] for a thorough treatment of the theory and implementation behind MinHash and LSH.

A Minhash is cheap to compute and has the following relation to the Jaccard similarity: The Jaccard similarity of two sets is the probability that their minhash is the same.
By computing many ($n$) minhashes of each name (second column in [@tbl:minhash]) we can approximate probability that the minhash of two sets is the same and thus their Jaccard similarity - as $n \to \infty$ this will become exact.
Next we use Locality Sensitive Hashing and group the $n$ MinHashes of each name into $b$ bands of size $r$ ($n = b r$) and checking for collisions in each band (third column [@tbl:minhash]) we identify names with a high probability of having the same MinHash.
Putting this all together, for each name in the first dataset of names (e.g. for each Glass name) we can efficiently identify names in the second dataset of names (e.g. Companies House names) that have a high probability of having the same MinHash and thus are likely to be highly Jaccard similar.

| Name           | $n=6$ Minhashes of Name | LSH Groups ($b=2, r=3$) |
| -------------- | ----------------------- | ----------------------- |
| “acme co“      | `[1,1,2,4,3,2]`         | `[112, 432]`            |
| “acme company“ | `[1,1,2,3,3,2]`         | `[112, 332]`            |
| “accenture“    | `[2,2,2,3,3,1]`         | `[222, 331]`            |

: Example of MinHash and LSH on three company names (first column). 6 different Minhashes are computed for each name (second column) and then grouped into two different bands and checked for collisions (third column). There is a hash collision between the first two names in the first bucket, therefore they are identified as similar. {#tbl:minhash}

### Chunked cosine similarity approach

We augment the pairs of likely similar names identified by the PDS approach with pairs identified by:
1. count-vectorising names to produce a matrix for each set of names (rows are names, columns are tokens);
2. {Term frequency}-{inverse document frequency} (TFIDF) transforming our matrices (considering both datasets of names to come from the same corpus for this purpose);
3. computing the cosine similarity of the TFIDF scores of company names and identifying the top $n$ most cosine similar pairs in the second dataset for each name in the first dataset.

This is performed in a computationally efficient manner by performing the dot-product between TFIDF matrices in a chunked manner.
<!-- (which can be efficiently computed and queried for the top $n$ most similar terms). -->

This provides a complementary approach to the PDS approach, which excels at capturing small character level discrepancies, by taking into account word ordering differences and information about the frequency of words across the corpus of company names.

### Computing exact similarities of subset

After identifying these sets of similar names we can apply the exact similarity measures that were computationally infeasible to naively apply across the whole dataset.

The similarity measures we chose to use for matching Glass to Companies House were:

- Jaccard Similarity of 3-shingles (exact)
- Cosine Similarity of TFIDF scores
- Levenshtein distance

### Choosing 'best' matches

After computing exact similarities for our subset of pairs, we choose the 'best' matches by identifying the Companies House name with the highest mean similarity score for each Glass name.
Each Glass organisation only appears once but each Companies House organisation may appear multiple times.
We only consider matches with a similarity score of 75% or higher - which we empirically determined as a sensible threshold.

### Criticisms

This matching is not exact and problems do exist. We have split the discussion of this matter into problems with the algorithm chosen to perform the matching and problems with the datasets themselves.

#### Problems with the algorithm

There are a number of hyper-parameters such as the shingle-size ($k$) and the number of MinHashes ($n$) to choose which may affect the accuracy of this approach.
In particular, choosing $n$ too low may introduce false negatives (we miss matching pairs); however larger values $n$ are prohibitive due to memory consumption.

Furthermore, choosing the best match based on the mean similarity is a fairly naive approach which one could replace with better heuristics that take into account factors such as string length, or other data such as the geographic proximity of two businesses (noting that there can be multiple Glass addresses for a website and that these are not guaranteed to match-up to the registered addresses present in Companies House).

#### Problems with the datasets

##### Glass

The business names in the Glass data are extracted from the text of websites and therefore are not guaranteed to be extracted correctly or even to correspond to the officially registered name of a company within Companies House.

##### Companies House

Due to the nature of Companies House, some matches may be to the wrong part of a conglomerate company which may have a different SIC designation.
Furthermore, many companies in Companies House have inaccurate SIC designations.
The IDBR team within the ONS have a modified version of the Companies House dataset which reassigns SIC codes and may contain information about company groupings.
Due to the timescales of this project it was not possible to access this data.


## Pre-processing of Glass business descriptions {#sec:glass_preproc}

The analysis of [@sec:topsbm] and [@sec:taxonomy] requires processing the raw descriptions extracted from business' websites by Glass into a form we can use to, for example, train a topic model on company descriptions or generate their vector representation to measure similarities between companies. We are specially interested in removing text which is uninformative about the industry where a company operates in but is likely to appear in a website, such as for example its location.

<!---
TODO: Add references to gensim and spacy?
--->

In order to do this, we build a Natural Language Processing (NLP) pipeline using the Spacy and Gensim Python libraries, which convert the raw 'string' of a business website description into an ordered list of 'tokens' where each token is a unigram, bigram or trigram composed of the lemmatised form of a word or an (uppercased) entity type label (for a subset of entity types) [@srinivasa2018natural, @vrehuuvrek2011gensim].

For example: `"I went to the Eiffel tower and visited my friend Jacques"` -> `["went", "GPE", "visit", "friend", "PERSON"
`]

Steps 1-8 are performed or rely on information extracted using the Spacy `en_core_web_lg` model, whilst steps 9-10 are performed or rely on information extracted using Gensim.

1. Tokenisation
4. Named entity recognition (NER) - Predict named entities using a transition-based method
<!-- TODO: summarise how these methods work? https://spacy.io/api/architectures#TransitionBasedParser -->
5. Lemmatise - Assign base forms to tokens
6. Merge entities - Merges series of tokens predicted by Spacy to be an entity into a single token
7. Filter
   - stopwords
   - punctuation
   - whitespace
8. Extract to list of strings - lemmatise or entity form
9. Generate n-grams - Use `gensim` to generate bi-grams, requiring that a bigram occurs at least 10 times and that the normalised pointwise mutual information (NPMI)[^NPMI] is 0.25 or higher.

10. Filter
   - short tokens
   - combinations of stop words
   - Words with low and very high frequency (those occurring less than 10 times and in more than 90% of documents)


[^NPMI]: NPMI $\in [-1, 1]$ where a value of: -1 indicates tokens never occur together; 0 indicates independence; and 1 indicates complete co-occurrence.

#### Token lemmatisation/remapping {#sec:remapping}

In step 8 tokens that are entities in the following categories are renamed to correspond to their entity category (uppercased):

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

We hypothesised that replacing these entities with their entity type name helps keep more information in the bag of words representation, particularly when entity types can be formed into n-grams with other words.

The alternative is that individual dates, people, organisation names etc. are too infrequent in the corpus to contribute information to the topic modelling approach.

Several entity categories such as `WORK_OF_ART`, `LANGUAGE`, `LAW`, `EVENT` were excluded as an empirical assessment of the classifications on sample business descriptions appeared inaccurate.

Furthermore, `PRODUCT` was left out because in this problem context (generating an industrial taxonomy) this is valuable information that we do not wish to homogenise.

Tokens that are not entities are replaced with their lowercase lemmatised form.
