# Topic modelling {#sec:topic_modelling}

## TopSBM pipeline {#sec:topsbm}

We train a TopSBM [@topSBM] topic model
 on our pre-processed collection of business descriptions.
This approach confers multiple advantages over the more traditional Latent Dirichlet Allocation (LDA) [@LDA] frequently used in the literature such as automatically selecting the number of topics; yielding a hierarchy of topics; permitting a more heterogeneous topic mixture than is permitted by LDA; and, crucially for the analysis of [@sec:similarities] generating document clusters.

RAM intensive => batch on subset of documents

### Model fit and hierarchy

| Model   |   Level |   Number of topics | Number of clusters |
|:--------|--------:|-----------:|-------------:|
| spacy   |       0 |        348 |          301 |
| spacy   |       1 |         68 |           54 |
| spacy   |       2 |         10 |           10 |
| spacy   |       3 |          3 |            3 |
: Number of topics and number of document clusters for top 4 levels of its hierarchy {#tbl:hierarchy}

| Model | $log(MDL)$ |
| ---- | ---- |
| Spacy | 16.80 |
: Log of the minimum description length (MDL) - lower is better. {#tbl:mdl}


| Topic 0   | Topic 1      | Topic 2      | Topic 3       | Topic 4     | Topic 5   | Topic 6    | Topic 7    | Topic 8      | Topic 9     |
|:----------|:-------------|:-------------|:--------------|:------------|:----------|:-----------|:-----------|:-------------|:------------|
| child     | GPE_GPE      | provide      | home          | service     | ORG       | vehicle    | building   | customer     | network     |
| school    | LOC          | client       | local         | business    | DATE      | car        | garden     | product      | financial   |
| community | people       | help         | event         | company     | GPE       | delivery   | clean      | supply       | agency      |
| course    | NORP         | team         | site          | project     | work      | stock      | cleaning   | market       | marketing   |
| learn     | staff        | support      | family        | industry    | PERSON    | shop       | kitchen    | equipment    | software    |
| student   | care         | experience   | PERSON_PERSON | solution    | offer     | item       | furniture  | brand        | insurance   |
| treatment | large        | range        | house         | system      | need      | hire       | energy     | job          | datum       |
| charity   | member       | quality      | produce       | property    | CARDINAL  | selection  | door       | installation | recruitment |
| patient   | want         | area         | facility      | development | include   | store      | electrical | supplier     | legal       |
| education | organisation | professional | serve         | require     | time      | collection | water      | repair       | finance     |
: Top 10 words for each topic at level 2 of the hierarchy {#tbl:topwords-spacy}


## SIC similarities {#sec:similarities}

### Level 0


![Level 0](figures/topsbm/SIC2_similarity_spacy-model_L0.png){#fig:spacy-L0 .altair}

At the most granular level of the cluster hierarchy...

Within section structure:

- Section A (Agriculture, Forestry and Fishing)
  - Low similarity except between 01 (Crop and animal production...) and 03 (Fishing...) in spacy model
- Section B: Mining and Quarrying
  - Block structure evident
  - Finer structure of similarity within section differs between models except between 06 (Extraction of petrol and gas) and 09 (Mining support service activities)
- Section C: Manufacturing
  - As a large section there are several sub-blocks evident
  - 32 (Other manufacturing) shares modestly high similarity across the full section
  - 13-15,31 (relating to textiles, clothes, and furniture) have obvious cohesive block structure
  - 10-12 (food, drink, tobacco) have some structure under the spacy model; however food is seen as more similar to tobacco than to drink!
  - 20,22-30,32-33 share a relatively cohesive block
  - 21 (Pharma manufacturing) is only similar to 21 (manufacture of chemicals) and 32 (other manufacturing)
- Section E: WATER SUPPLY; SEWERAGE, WASTE MANAGEMENT AND REMEDIATION ACTIVITIES
  - Weak block structure
  - 36 (water collection) less similar to other divisions
- Section F: CONSTRUCTION
  - Very strong block structure captured
- Section G: WHOLESALE AND RETAIL TRADE; REPAIR OF MOTOR VEHICLES AND MOTORCYCLES
  - 46 (wholesale trade) and 47 (retail trade) very similar
  - 45 (wholesale trade and retail trade of motor vehicles) distinct from 46 and 47
- Section H: TRANSPORTATION AND STORAGE
  - Very strong block structure across section
- Section I: ACCOMMODATION AND FOOD SERVICE ACTIVITIES
  - Very strong block structure across section
- Section J: INFORMATION AND COMMUNICATION
  - Obvious mixed block structure
  - Telecomms (61), IT services (63), and computer programming (62) cluster strongly together
  - 59 and 60 cluster together
- Section K: FINANCIAL AND INSURANCE ACTIVITIES
  - Strong clustering
- Section M: PROFESSIONAL, SCIENTIFIC AND TECHNICAL ACTIVITIES
  - 70 (Head offices / management consultancy) and 74 (Other activities) are highly similarly to other divisions within the section but otherwise similarity is low
- Section N: ADMINISTRATIVE AND SUPPORT SERVICE ACTIVITIES
  - Little block structure evident, 82 (Office admin/support) is moderately similar to other divisions within the section but no other division is similar to another
- Section Q: HUMAN HEALTH AND SOCIAL WORK ACTIVITIES
  - Strong clustering
- Section R: ARTS, ENTERTAINMENT AND RECREATION
  - Some divisions show similarity but block structure is weak (particularly under the simple model)
- Section S: OTHER SERVICE ACTIVITIES
  - Very weak similarity. Expected as divisions within this section are not obviously related.
- Section T: ACTIVITIES OF HOUSEHOLDS AS EMPLOYERS; UNDIFFERENTIATED GOODS- AND SERVICES-PRODUCING ACTIVITIES OF HOUSEHOLDS FOR OWN USE
  - Very weak similarity between the two divisions

Selection of interesting cross-section relationships:

- 01 (Crop and animal production, hunting...) and 03 (Fishing...) are highly related to 55 (Accomodation) and to a lesser extent 56 (Food and beverage services)
- Mining and quarrying are weakly similar to various Manufacturing activities, in particular 08 (Other mining/quarrying) is highly similar to 23 (Manufacture non-metallic minerals)
- Manufacturing section is similar to 46 and 47 (wholesale and retail trade)
- Section J (ICT) and Section M (Professional, Scientific, Technical activities) share high similarity
- 77 (Rental and leasing) is similar to divisions in sections G (trade) and H (Transport and storage). This is more strongly picked up by the spacy model.
- 82 (Office support/admin) is similar to many divisions across all sections
- Divison 99 which includes unclassified organisations as well as extraterritorial organisations and bodies has high simiarities to 82 (Office admin/support), 74 (Other Professional, Scientific, technical) activities, 46 and 47 (trade), 32 (other manufacturing) and others. Similarities here could be an indication about what types of businesses are misclassified.

### Level 1

Level 1 shows a stronger structure than that of level 0.
Furthermore, the two models begin to exhibit different behaviour from one another.

Some examples of extra structure that is revealed:

- Particularly under the spacy model, 45 (motor vehicle trade) is highly similar to section H (transportation and storage).
- Under the spacy model Section L (Real estate) is highly similar to Section K (Financial and insurance activities)
- Under the spacy model (and to a much lesser extent the simple model), Sections C,D,E,F share much more structure.
- Divisions 10-12 (Food, drink, tobacco) are much more strongly related to 56 (Food and beverage services) in particular
- Section B (mining and quarrying) has high degrees of similarity with secondary sector sections as well as section M (Scientific and technical activities).
- The spacy model picks up a high similarity: between 72 (Scientific R&D) and 21 (Pharma manufacture); between 73 (Advertising) and 18 (Printing and reproduction of recorded media)
- The simple model picks up a high similarity between 49 (Land transport) and 79 (Travel and tour activities)

![SIC similarities (Level 1)](figures/topsbm/SIC2_similarity_spacy-model_L1.png){#fig:spacy-L1 .altair}

### Level 2

The level 1 spacy model structure looks similar to a superposition of the simple model's level 1 and level 2 structure, hinting at a possible reason for some of the differences seen at level 1.
Continuing in the spirit of level 1, the spacy model exhibits a higher level of structure again.

The general observations seen at level 0 and 1 remain relevant; however the connectedness of sections is much higher.

Some interesting differences between the two models become more apparent now:

- Under the spacy model, 45-53 (trade, transportation and storage) are highly similar to 10-15 (manufacture of food, drinks and clothes) and moderately similar to other secondary sector activities. 
  However, under the simple model, 45-53 are much more similar to other secondary sector activities and some primary sector activities such as mining and quarrying than they are to 10-15.
  Common sense falls quite strongly on the side of the spacy model here.
- Under the spacy model Sections J, K, L, M (broadly high-paid tertiary-sector jobs) are highly similar to Section B (mining and quarrying) and 17-18 (manufacture of paper, printing, and recorded media). The first of these is peculiar and perhaps and artifact of the type of mining and quarrying companies which may have a web presence. The relation to 17-18 is highly logical.
- Interestingly the spacy model picks up 75 (Veterinary activities) as highly similar to Section Q (Human health and social work activities)



![SIC similarities (Level 2)](figures/topsbm/SIC2_similarity_spacy-model_L2.png){#fig:spacy-L2 .altair}


## SIC heterogeneity

By aggregating the topic distributions by SIC, and calculating the entropy of the topic distributions for each SIC, we get one measure of the "heterogeneity" of sectors within the Glass data.

![SIC heterogeneity at SIC division level](figures/topsbm/SIC2_entropy.png){#fig:hetero-division .altair}

<!-- ![SIC heterogeneity at SIC group level](figures/topsbm/SIC3_entropy.png){#fig:hetero-group .altair} -->

![SIC heterogeneity at SIC class level](figures/topsbm/SIC4_entropy.png){#fig:hetero-class .altair}

![Topic distribution amongst the top 10 and bottom 10 most heterogeneous (according to entropy of the topic distributions with SIC codes) sectors. Based on level 1 of the model hierarchy.](figures/topsbm/SIC2_entropy_topicdist_L1.png){#fig:hetero-topic-dist-division .altair}

![Topic distribution amongst the top 10 and bottom 10 most heterogeneous (according to entropy of the topic distributions with SIC codes) sectors. Based on level 1 of the model hierarchy.](figures/topsbm/SIC4_entropy_topicdist_L1.png){#fig:hetero-topic-dist-class .altair}

# Appendix
