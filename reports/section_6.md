# Conclusion

Our analysis of a labelled dataset of SIC codes and company descriptions obtained from business websites has helped to assess the limitations of the current SIC taxonomy, identify opportunities to improve it using natural language processing and network analysis and some of the new challenges created by this approach. Here we summarise key issues, discuss next steps for the research and highlight broader implementation factors.

## Limitations of the current taxonomy

Our analysis of the SIC taxonomy using supervised and unsupervised methods in Sections 3 and 4 highlights important mismatches between the language that businesses use to describe what they do and the SIC codes where they are classified. More specifically, we find that businesses with similar descriptions are sometimes placed in different SIC codes, and that businesses with different descriptions are sometimes placed in the same codes. Our decomposition of sectors into highly grained communities illustrates the degree of heterogeneity in SIC codes such as 7490, which appears to include companies providing support services to the pharmaceutical industry, companies working with renewable energies and specialist lawyers. 

The resulting misclassification is likely to introduce noise into economic indicators about the composition of the economy specially when those are produced at a high level of industrial resolution. While we are aware that some of the sectoral misclassification present in Companies House is likely to be addressed 'downstream' as additional data is collected from businesses, it is unclear how this might help to address situations where businesses are active in areas currently absent from the SIC taxonomy, or where there is ambiguity about their industrial focus because they operate in multiple sectors. The bottom-up taxonomy that we have piloted in Section 5 sets out to overcome some of these limitations.

## Advantages of a new, bottom-up taxonomy

Section 5 illustrates some of the advantages of a new taxonomy based on semantic clustering of companies based on their text descriptions: new ecomomic activities - for example around the green economy - can be detected and studied, and companies can be labelled with multiple sectors thus capturing their diversification or the adoption of particular technologies and practices. We are able to 'open up' the black box of "not elsewhere classified" SIC codes and analyse their composition. 

A bottom-up industrial taxonomy may also make it possible to look at the economy from new and potentially useful perspectives e.g. sets of companies that are part of the same value chain or have adopted similar technologies or production processes complementing those offered by the SIC taxonomy.

## Challenges for developing a new taxonomy

Section 5 also shows the challenges for developing a bottom-up industrial taxonomy: this involves a complex pipeline with multiple steps and some hard to interpret results. Our decision to focus on companies with a stronger presence in the Glass data  leads to the removal of SIC codes in primary sectors that are less well-covered there. These limitations point at opportunities for improvement and further development that we will pursue in the next phase of the project.

## Next steps

As noted, our next step is to simplify and improve our text processing, sector identification and company classification pipeline, and to use the results in an applied analysis that demonstrates the value added of the taxonomy. Some options for such analysis include:

* Analysing the economic geography and performance of a new and policy relevant sectors such as for example the green economy, that are currently absent from the SIC taxonomy.
* Analysing growth dynamics in bottom-up sectors of interest after matching a firm-level dataset labelled with new sectors and micro surveys such as IDBR, ABS or the Community Innovation Survey
* Implementing an alternative, hierarchical bottom-up taxonomy and comparing its geography and evolution with SIC-2007. This will require developing methods that draw on official, comprehensive micro-data to estimate indicators of economic performance such as employment in new sectors.

## Implementation considerations

We conclude by highlighting additional considerations for the implementation of a bottom-up industrial taxonomy based on company descriptions fron their websites:

1. Coverage: As we previously pointed out, business websites may fail to cover some industries, which could create gaps in the taxonomy. One way to address this would be to deploy the bottom up industrial taxonomy as a tool to improve the resolution and granularity of analysis of knowledge intensive activities which our analysis suggest are particularly poorly-served by the industrial taxonomy, while preserving existing codes for other sectors.
2. Estimating employment levels: Business websites offer rich information about the markets that a business serves but miss important dimensions of economic activity such as employment or productivity. One way to address this gap is by matching company descriptions and sectors with official micro-data  such as for example IDBR or ABS drawing on Companies House numbers we have obtained through the fuzzy matching protocol outlined in Section 2. An additional step would be required to estimate population level statistics in various sectors from the incomplete sample of companies we have access to.
3. Longitudinal updates: A bottom-up industrial taxonomy could in principle be updated close to real-time as the economy evolves and new industries appear and are reflected in company descriptions. While this could offer a very timely perspective on the composition of the economy, it would come at the cost of longitudinal consistency in terms of our ability to study the evolution of industries over time. One potential strategy to manage this trade-off would be to maintain a frequently updated bottom-up industrial taxonomy at the sub-SIC4 level with less regular updates above that. One advantage of this approach is that it would help to identify nascent sectors with a critical mass of activity that might warrant the creation of higher-level codes.

These considerations suggest that in order to benefit from novel data sources and methods, the economic statistics system will need to innovate in the infrastructures it uses to connect open and web sources with official microdata, to monitor the evolution of the economy in order to identify the emergence of new sectors close to real time, and to adopt machine learning methods in order to transform that enhanced understanding into up-to-data industrial taxonomies offering the granular and timely views of the economy and its constituent industries that policymakers increasingly demand.


# Bibliography