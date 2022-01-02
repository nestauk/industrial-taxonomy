
# Introduction {#sec:introduction}

The Standard Industrial Classification (SIC) provides an organising framework for the analysis of the sectoral composition of the UK economy [@hughes2009implementation]. It consists of a hierarchy of industrial codes that describe the economic activities of firms at increasing levels of resolution (see @tbl:1 for an example).

|Section     | Division      | Group        | Class        |
|------------|---------------|--------------|------------ |
|Financial and Insurance Activities (K) | Financial service activities, except insurance and pension funding (64) | Insurance (651) | Life insurance (6511)|
: Example of the structure of the SIC taxonomy {#tbl:1}

Companies self-select their SIC code when they register with Companies House. Subsequently, they may be reassigned into a new code if, for example, an error is identified when the company undertakes an official business survey.

Eventually, these classifications are used to produce official statistics about the business population and the sectoral distribution of productivity, employment, occupations and salaries through a variety of ONS products such as the Interdepartmental Business Register (IDBR), the Annual Business Survey (ABS), the Business Register Employment Survey (BRES), the Annual Population Survey (APS) or the Annual Survey of Hours and Earnings (ASHE). 

The widespread use of the SIC taxonomy to produce 'sectoral cuts' of other economic statistics underscores its importance: different sectors vary in their productivity, geography, skills needs, innovation activities, business models and internationalisation among other factors so understanding their levels of activity and how it evolves over time is critical for informing a host of economic policies. The increasing importance of industrial policy, regional rebalancing ('levelling up') and sustainability policy agendas places an additional premium on access to timely and granular statistics about the industrial composition of the UK and its national, regional and local economies.

There is increasing awareness of the limitations in the version of the SIC taxonomy currently in use which makes it less relevant for these policy needs [@hicks2011structural]. They include:

* Lack of timeliness: The version of the SIC taxonomy currently in use was last updated in 2007, making it unsuitable for the analysis of industries that have emerged since then and are of particular interest for policymakers such as Artificial Intelligence, Fintech, Renewables and the "Gig economy".
* Presence of uninformative sectors: Out of the ca. 600 four-digit SIC codes ('classes') in the SIC taxonomy, 52 refer to 'other activities' or "activities not elsewhere classified", employing 15% of the workforce in 2019 according to BRES. 
* Difficulties accommodating companies that straddle sectors: the SIC taxonomy is completely exhaustive and mutually exclusive, meaning that all companies are classified in a code and each  company is classified in a single code. This has the advantage of avoiding double counting but might create challenges classifying business that undertake activities captured in several SIC codes, such as for example a fintech company that applies digital technologies (captured in Division 62:Computer programming, consultancy and related activities) in financial services (captured in Division 64:Financial service activities, except insurance and pension funding).^[Here, it is worth noting that businesses can select multiple SIC codes when becoming incorporated but it is unclear how often they do this, and information about business secondary codes is rarely reported.]
* Misclassification: Together, all the reasons above lead to concerns that companies might select the wrong code because they do not see their activities reflected in the current taxonomy or opt for a "not elsewhere classfied code" even when a suitable code is available somewhere in the SIC taxonomy.^[As we noted previously some of these instances of misclassification may be rectified subsequently when additional data is collected, for example through a business survey.]

The increasing availability of open and web data about what companies do (or say they do) provide some interesting opportunities to address some of these limitations in the current SIC taxonomy [@bean2016independent]. 

In particular, web sources such as business websites and sector-specific directories and web portals have been used to measure the digital economy [@nathan2015mapping], the video-games sector [@mateos2014map], the 'immersive economy' (including technologies such as Virtual Reality and Augmented Reality) [@mateos2018immersive] and industrial clusters in the UK [@nathan2017industrial]. In a previous analysis, we used business websites to map the geography of businesses using emerging technologies in the UK [@bishop2019exploring]. 

This approach has the advantage of relying on what businesses say they do in order to reach their customers instead of their engagement with the administrative process of selecting a code when they become incorporated. We would expect these statements to be more timely than the codes in the SIC-2007 taxonomy insofar businesses have incentives to keep their websites updated, and to mention new products, services, processes and technologies that may be of interest to their customers, thus capturing emerging industries. They should also reflect the multiple economic activities that businesses engage in regardless of whether they are confined to a single industrial code.

Unstructured descriptions of business activities are however not without their limitations. Novel sources may capture unrepresentative samples of the business population, and business descriptions are likely to be noisy, in part because they serve the purpose of promoting products and services and attract new customers rather than providing an accurate description of what businesses do. There is also the significant challenge of transforming all this unstructured information into a taxonomy that can be used to measure economic activities in different industries and eventually make policy-relevant statements about the composition of the economy.

In this working paper we report emerging findings from an exploration of the opportunities and challenges for building a bottom-up industrial taxonomy based on text data complementing the SIC taxonomy and addressing some of its limitations. 

In order to do this, we match a dataset of business website descriptions obtained from Glass, a business intelligence startup, with Companies House. We then assess the alignment between SIC codes at the four-digit level and business descriptions using supervised and unsupervised machine learning methods, and pilot an approach to build a bottom-up industrial taxonomy based on an analysis of the text in company descriptions. 

The structure for the report is thus:

Section 2 introduces our data and how we have processed it with a special focus on the fuzzy matching algorithm we have developed in order to combine business website descriptions with Companies House and the natural language processing pipeline we use to process company descriptions.

Section 3 presents the results of a supervised machine learning analysis where we train a predictive model on our labelled dataset in order to determine the extent to which it is possible to predict 4-digit SIC codes using the text in business descriptions, and the explanation for various instances of misclassification.

Section 4 presents the results of an unsupervised machine learning analysis where we train a hierarchical topic model on our dataset with the goal of assessing the semantic homogeneity / heterogeneity of 4-digit SIC codes (i.e. the extent to which they contain companies with widely varying descriptions of their activities) as well as semantic overlaps between codes in different parts of the SIC taxonomy.

Having 'dissected' the SIC taxonomy using machine learning methods and diagnosed some of its limitations, Section 5 trials an experimental, iterative strategy to build a bottom-up industrial taxonomy that addresses some of them. In order to do this, we use a network-based topic model to decompose 4-digit SIC codes into more granular text sectors. We explore the opportunities that this opens up for decomposing uninformative SIC codes into their constituent parts, studying policy-relevant "sectors" such as the green economy, characterising more accurately the composition of local economies, and clustering text sectors into a hierarchical industrial taxonomy.

Section 6 presents conclusions and next steps.
