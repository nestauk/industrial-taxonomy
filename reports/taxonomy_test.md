# 3. Building a bottom-up industrial taxonomy
In this section we summarise the results of an initial exploration of an approach to build a bottom-up industrial taxonomy using business descriptions from websites. It involves the following steps:

1. Extracting keywords and keyphrases (KW/KP) from business descriptions
2. Creating KW/KP networks based on their co-occurrence in business descriptions
3. Decomposing these networks into communities of frequently co-occurring KW/KPs that may capture industries

We have tested this approach with samples of companies from two 4-digit SIC codes. They are:

* 6201: Computer Programming Activities
* 7490: Other Professional, Scientific And Technical Activities Not Elsewhere Classified

We have selected these two SIC codes because they illustrate two use cases where a bottom-up taxonomy may be particularly valuable: identifying sub-sectors that are developing or adopting emerging technologies and operating in new markets (in `6201`) and decomposing uninformative "Not elsewhere classified" sectors into meaningful subsectors.

We outline the methodology in subsection 1 and emerging finding in subsection 2. Subsection 3 discusses limitations and next steps.

## 1.  Methodology

We (pragmatically) define a sector as a set of companies that use similar processes to produce goods and services for similar markets. One way to identify such sectors in a 'bottom-up', empirical way, is to look for co-occurrences of terms referring to production processes, goods and services in the same company description. Doing this requires building a vocabulary of relevant terms.

We have explored two avenues to do this:

### a. Top-down approach: United Nations Standard Product and Service Classification (UNSPSC)

The UNSPSC is a regularly updated taxonomy of products and services maintained by the UN. It contains 4 levels of aggregation (_segment_, _family_, _class_ and _commodity_), with just under 148,000 commodities at the highest level of resolution. 

In principle, this taxonomy offers a principled and interpretable approach to identify mentions to products and services in business descriptions that we can then use to build our co-occurrence network. Unfortunately, we find that the terms used in UNSPSC have low levels of overlap with the vocabulary in business descriptions. The main reason for this is that UNSPSC terms are often long, detailed and specific (see table 1 for some random examples).

|Segment | Commodity|
|---------|---------|
| Live Plant and Animal Material and Accessories and Supplies | Dried cut white snapdragon |
| Food Beverage and Tobacco Products | Canned or jarred kapia peppers |
| Power Generation and Distribution Machinery and Accessories | Aluminum triplex service drop cable |
| Drugs and Pharmaceutical Products | Tizanidine |
| Healthcare Services | Drainage of right spermatic cord with drainage device,  percutaneous approach |
| Healthcare Services | Bypass right kidney pelvis to ileocutaneous with autologous tissue substitute,  open approach |
| Food Beverage and Tobacco Products | Canned or jarred summer blush nectarines |
| Healthcare Services | Dilation of right pulmonary vein with intraluminal device,  open approach |
| Healthcare Services | Reposition right lower femur with internal fixation device,  percutaneous endoscopic approach |
| Healthcare Services | Drainage of right ethmoid bone with drainage device,  percutaneous endoscopic approach |

It is unlikely that firms selling these goods and services would mention them using the same language as UNSPSC in the business descriptions that we have access to. The table illustrates that the taxonomy is highly unbalanced, with one of the families (_Healthcare Services_) comprising 73,000 commodities. We also find that the taxonomy does not include terms related to emerging technologies such as _Artificial Intelligence_ or _Machine Learning_, an aspect of company activities that we are specially interested in capturing through our analysis.

### b. Bottom-up approach: Automated keyword and keyphrase extraction methods

An alternative strategy is to exploit information about word positions in company descriptions in order to build sets of keywords and keyphrases (sets of frequently occurring words such as _web developer_ or _building surveyor_) that we can then use to build our co-occurrence network. As part of this test, we have explored three algorithms to extract these keywords / keyphrases (KW/KPs).

* `RAKE` (Rapid Automatic Keyword Extractor) is an algorithm that identifies frequently occurring sets of words in documents that are not uninformative stopwords or separated by punctuation (Rose et al, 2010).
* `YAKE` (Yet Another Keyword Extractor) exploits further patterns such as the position of a candidate KW/KP in a document (upweighting those that appear at the beginning) or the range of sentences where a candidate KW/KP occurs (Campos et al 2018).
* `KeyBERT` uses a pre-trained language model to identify KW/KPs in the document that are closest to it in vector space (more semantically similar / better able to summarie the document) (Sharma and Li, 2019).

We note that all these algorithms operate at the document rather than corpus level. One potential avenue for further exploration is to adopt an ngram extraction activity that identifies combinations of words that happen across the corpus at a higher rate than would be expected if they were statistically independent.

We use open source implementations of these three algorithms in order to test their suitability. Next section outlines our pipeline for this.

### c. Pipeline

[pipeline]("./figures/extract_pipeline.png")





