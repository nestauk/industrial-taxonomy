---
title: "Using text data to improve industrial statistics in the UK"
subtitle: "An exploratory study"
author:
    - "Alex Bishop"
    - "Juan Mateos-Garcia"
    - "George Richardson"
date:
    - 11 May 2021
figPrefix:
  - "figure"
  - "figures"
tblPrefix:
  - "table"
  - "tables"
secPrefix:
  - "section"
  - "sections"
bibliography: /bibliography.bib
toc-depth: 2
toc: true
secnumdepth: 3
---

# Executive summary {-}

A growing number of economic policy agendas in the UK demand accurate, detailed and timely statistics about the industrial composition of the economy. This requires an industrial taxonomy that reflects existing industries and emerging ones enabling accurate classification of companies into sectors to produce economic indicators that can be used to inform policy. 

The current version of the Standard Industrial Classification (SIC) that serves this purpose has some important limitations that could limit its usefulness. They include lack of timeliness (the taxonomy was last updated in 2007), presence of uninformative codes and difficulties accommodating companies that operate across multiple sectors.

In this report we use novel data sources and state-of-the-art machine learning methods to evidence the limitations of this taxonomy and explore options to develop a complementary taxonomy overcoming those limitations. In order to do this, we:

1. Match 1.8 million business website descriptions procured from Glass, a big data business intelligence company, with Companies House in order to label these descriptions with the SIC codes that these companies selected when they became incorporated
2. Use pre-trained text classification models to analyse the extent to which it is possible to predict companies' SIC codes based on their description, and potential explanations for model error that might be suggestive of limitations in the SIC taxonomy.
3. Use hierarchical topic modelling to characterise the homogeneity or heterogeneity of different SIC codes potentially helping us to identify codes that are particularly suitable candidates for decomposition into a more granular set of industries in a bottom-up, data-driven way
4. Pilot a strategy to build such a bottom-up industrial taxonomy by using the text of company descriptions to cluster them into 'text sectors'.

The results confirm our priors about the limitations of the SIC taxonomy currently in use: we identify important mismatches between company descriptions and the SIC4 codes where they are classified, and demonstrate the heterogeneity of "other activities not elsewhere classified" codes that in some cases contain activities ranging from plumbing to social services and religious activities (in SIC4 8299) and from legal services to the organisation of clinical trials and the supply of renewable energies (in SIC4 7490). 

Our emerging results also illustrate the potential of a bottom-up industrial taxonomy for generating sectoral categories that can be used to measure notable economic activities for example related to sustainability and the green economy. Some limitations of our approach include insufficient coverage of sectors outside the 'knowledge economy' and the presence of some noisy sectors in the data.

We conclude by setting out potential avenues to overcome these limitations and deploy, in a forthcoming ESCoE project, a bottom-up industrial taxonomy to analyse the UK economy in a way that demonstrates the value added of the methodology we have piloted here.
