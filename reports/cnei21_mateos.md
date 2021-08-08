---
title: "Discovering industries in networks of words"
date: 5 May 2021
fontsize: 10.5pt
geometry: margin=3.5cm
author:
    - "Alex Bishop"
    - "Juan Mateos-Garcia"
    - "George Richardson"
---
# Abstract

The standard industrial classifications used to study the composition of the economy suffer from well-known limitations such as lack of granularity and timeliness, presence of heterogeneous and uninformative industry codes and difficulties accommodating businesses whose activities straddle sectors. These features simplify economic analysis while reducing our ability to study the emergence of new industries and the transformation of existing ones. 

Here, we use machine learning and network science to analyse business descriptions extracted from their websites in order to generate an alternative, bottom-up industrial taxonomy where we define industries as communities within networks of words.

Our starting point are 1.5 million business websites in the UK that we match with the business register in order to produce a labelled dataset of industrial codes, business descriptions and additional metadata such as location and incorporation date. Having evidenced the limitations of the current UK industrial taxonomy (SIC-2007) through supervised and unsupervised analyses of this labelled dataset, we proceed to construct a bottom up-taxonomy whose building blocks are communities of terms (ngrams) that co-occur in company descriptions. 

This involves implementing a natural language processing pipeline to remove generic and promotional terms from company descriptions, building and simplifying a co-occurrence network _inside_ each 4-digit SIC code and decomposing this network into its constituent communities of terms (_new sectors_) using an ensemble of algorithms implemented in the Python `cdlib` library (see figure next page for a visualisation of the term graph and its community partitions for SIC4 6201: Computer Programming Activities at the top and SIC 9609: Other Personal Services Not Elsewhere Classified at the bottom).

Having done this, we tag companies with those new sectors where there is a strong overlap between their descriptions and new sector vocabularies, and analyse their geography paying special attention to community combinations that might tell us something about technology adoption (e.g. presence of non-digital and digital communities inside company descriptions) and position in the value chain. We also use hierarchical clustering to reconstruct the hierarchy of a bottom-up industrial taxonomy based on sector co-occurrences in companies and compare its structure with the top-down SIC industrial taxonomy.

Our results illustrate the potential for deploying new data sources and network science methods to create representations of the economy that reflect more accurately its complex and evolving structure.

![Word (ngram) co-occurrence network and community partitions for SIC4 6201: Computer Programming Activities (Top) and SIC 9609: Other Personal Services Not Elsewhere Classified (Bottom). Each node is an n-gram, the connectionss between them are their co-occurrences in company descriptions. The colour of the nodes represents their communities. The graphs have been simplified using a maximum spanning tree algorithm.](images_paper/network_comp.png)