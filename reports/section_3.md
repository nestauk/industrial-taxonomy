# Section 3: Using predictive modelling to analyse a taxonomy

How well can a machine learn the relationship between what a company says it does and the SIC code that company has been assigned? In this section, we attempt to answer this question by training a machine learning model to predict 4-digit SIC codes based on company descriptions and analysing the results. We do this in order to better understand the challenge of fitting companies into the SIC-2007 taxonomy based on the activities that they use to describe themselves.

To do this, we have taken advantage of recent advances in machine learning, in particular neural network architectures known as transformers, which are particularly well-suited to natural language tasks. Transformers are capable of modelling linguistic relationships in ways that have made them suitable for research and applications in text classification, question and answering, text summarisation, machine translation and text generation. They have been rapidly adopted by the open source community, with tools such as the Transformers Python library allowing developers and researchers to easily obtain and use models for various natural language processing (NLP) tasks [@wolf-etal-2020-transformers]. Importantly for this work, the library provides access to *pre-trained* models, which have already been trained on a large corpus of text to recognise language patterns and can subsequently be adapted for new tasks with relatively little data. Here we describe we describe first how we have used such a model to predict SIC codes and second, how we have used machine learning performance metrics to understand where the model succeeds and struggles.

## 2.1 Modelling methodology

### Transformers

The model we build is a multi-class classifier that attempts to predict one of the 615 4-digit SIC codes, using the description of a company as its input. The development of the classifier is treated as a supervised machine learning problem - one in which a labelled dataset is used to train a model. As we have described above, there are no official datasets that contain company descriptions and their associated SIC codes. In this case, we have used the fuzzy matched data from Glass AI and Companies House to obtain a labelled dataset where SIC codes are associated with business descriptions.

We use a transformer model to construct the classifier. Transformers and other neural networks are in large part able to learn complex linguistic relationships because of the large numbers of parameters that they contain. These parameters are the mathematical weights and biases that describe the relationships between different nodes inside the network and that ultimately determine the relationship between the inputs of a model and its outputs. Transformers can have many millions or billions of parameters to tune, which in turn requires large volumes of data and ultimately means that even with optimised software and hardware, training a model is time and energy intensive.

Fortunately, researchers have discovered that once a model has been trained on a large volume of data for a particular task, the patterns that it learns are often highly generalisable. A transformer that is trained on a large corpus, such as the text from English language Wikipedia, for one task will result in a model with parameter values that represent a large number and diversity of patterns that are valuable for other NLP tasks. Because of this, training a model from scratch is often unnecessary and instead, developers can opt to "fine-tune" an existing model for their specific problem.

The Transformers library makes this particularly easy by offering a consistent tool for working with models built in different frameworks (TensorFlow and PyTorch) and with different model architectures, enabling access to models that have been pre-trained on large datasets with accelerated hardware. This allows community members to share models with each other, reducing the amount of collective time and resources that must be spent on model training. It also provides an interface for fine-tuning those models. 

For the SIC code classifier, we fine-tuned an English language DistilBERT model. This is a transformer that has a reduced number of parameters to reduce disk space, memory use and training times, but that still performs comparably to other larger models. The model was trained on a masked language modeling task where 15% of words in a sentence are randomly masked and the model must predict what those masked words are based on the unmasked content. This gives the model an inner representation of the English language that is then transferrable to a classification process. The additional component to turn the model into a classifier is the addition of a fully connected, final layer to the neural network that has a number of nodes equal to the number of classes being predicted. The entire model is then trained by feeding it the text examples in a dataset, requiring it to reduce the loss between the predictions made in the final layer and the labels in the dataset. This process tweaks the values of all the parameters in the network, as well as the final classification layer.

### Implementation

In practice, several design choices were made in the development of this model. The first is that a higher match threshold between the Glass AI and Companies House datasets was chosen, compared to some previous work in the Data Analytics team. In an initial prototyping of the model to predict the highest level of the SIC index (the Section), it was noticed that predictions on company descriptions with a higher matching score yielded a higher F1 score. The micro F1 score peaked when the match threshold was around 75, therefore only company descriptions that matched to Companies House with a minimum score of 75 were chosen to train, validate and test the model. In total, after the matching threshold had been applied, there were around 350,000 samples for training. In this phase of development, 80% of these were used for training, while 20% were held out as the test set for analysis.

Second, an optimisation technique was applied to mitigate a limitation of transformers that can slow down the training time for the fine-tuning process. Transformers are trained by passing data through the model in batches. Several examples of text are shown to the model before its predictions are evaluated against the true labels and the parameters are updated. In each batch, the number of input features - in this case, the input features are the tokens in the company descriptions -  must be consistent across all samples.  Because of this, shorter texts in a batch are padded up to the maximum length with a filler token that has no bearing on the model parameters.

In addition, transformers also have a maximum input size (for DistilBERT this is 512 tokens). Any examples in a batch can only be padded up to this length or a shorter length specified by the user. Any sequences longer than the specified maximum or upper limit are truncated. This is relevant to the training speed as the time it takes for a transformer to process a piece of data is quadratic with its length. For any training process with variable length texts there is always the challenge of setting a maximum length that captures sufficient volumes of information in longer sequences and that does not impose a significant speed penalty. Fortunately, the batching process permits the use of dynamic padding and uniform batching, two techniques that when combined can offer a significant speed up.

Dynamic padding is the process of applying tokenisation at the point of batching the data, rather than on all data at the start of the model development process. In this way, a maximum length can be chosen that is no longer than the longest sequence in a batch. This helps by ensuring that no examples are padded more than they need to be. A global maximum can still be set to ensure no sample is padded beyond a certain length.

Uniform batching is the approach of grouping texts of similar length together before they are passed to the transformer in batches. In this way, the amount of padding needed in a given batch is reduced yet again. One way of doing this is by sorting the data by length to ensure that shorter texts are found close to each other. A schematic for dynamic padding and uniform batching is shown in [@fig:dynamic_padding_uniform_batching]. Both methods were applied to the data during the training process for the SIC classifier and resulted in significant speedups.

![With dynamic padding and uniform batching, texts of similar length are grouped together and padded only to the length of the longest sequence, ensuring that all batchs, and therefore all sequences, take a minimal time to process.](tables_figures/dynamic_padding_uniform_batching.png){#fig:dynamic_padding_uniform_batching}

After training the model by following the steps above, the classifier was applied to the remaining 20% of company descriptions in the test dataset. These predictions form the basis of the analysis in the following section.

It is worth noting that the process described here does encompass some limitations, particularly in relation to the training data. The first is that the dataset has a high class imbalance. There are some SIC codes with many thousands of examples and others with only a handful. This means that the information available to the model for learning some SIC codes is significantly larger than for others. A second limitation is that we cannot consider the labelled dataset as a gold standard, as manual inspection of the data shows that in some cases the label associated with a company description does not constitute a reasonable fit. In yet other cases, the description provided is not adequate to easily determine a single suitable label out of the 615 available, as a combination of codes would more suitably describe some organisation's activities. However, in some ways it is these aspects that make this analysis interesting.

## 2.2 Model performance and analysis

Evaluating a model with 615 possible output classes is not a trivial task. Machine learning often involves making performance tradeoffs both within and between classes and therefore the number of possible classes increases the complexity of this task by making the final optimisation objective more difficult to define. In this analysis we do not attempt to comprehensively determine the efficacy of the model, but rather make use of various metrics typically used within model evaluation to highlight findings that are relevant to the limitations of the SIC-2007 taxonomy and the creation of a data driven alternative.

The first observation is that the classifier only makes predictions for 95 of the possible 4-digit SIC codes, despite the training data covering companies from all of the 615 codes. Classification metrics, such as precision, recall and F1 score are therefore only available for a subset of the codes, however we do shed some light on why this extreme aspect of low performance occurs by combining these with other metrics. In [@fig:sic4_f1] we see the most performant SIC codes according to their F1 score. We can see that there are only 6 codes with scores above 0.7, before a dropoff and gradual decline in scores. From a manual inspection of the results, there appear to be no obvious characteristics of the codes, such as industry type, that determines their performance, however it is worth noting that there are very few low-specificity codes (such as those that are defined by their inclusion of n.e.c. - not elsewhere classified) among the most performant. As we will see, the results are more nuanced than this. Overall, this highlights that the model struggles to assign companies a single label succesfully according to the test data.

![Top 4-digit SIC codes according to their F1 scores.](tables_figures/sic4_vs_f1-score_barh.png){#fig:sic4_f1}

While this gives an overall impression of accuracy, it does not tell us much about the nature of the misclassifications. The set of companies with a particular SIC code may be misclassified into one other class or many, and these may relate closely to the original category or be very distant in terms of industrial activity. In order to determine the diversity of missclassifications for companies in each SIC code and to understand why this might be happening, we calculate two additional metrics: the Shannon index and the Silhouette score.

The Shannon index is a measure of entropy, often used to describe the diversity of states in a system. For companies in the test set with a SIC code, $i$, it is defined as

$$ H_{i} = -\sum_{j=1}^s\left(p_j\log_2 p_j\right) $$ {#eq:shannon}

where $s$ is the number of operational taxonomic units (in this case 615) and $p_{j}$ is the proportion of companies classified with SIC code $j$. This tells us the degree to which companies are misclassified into the possible codes.

Second, we calculate the silhouette coefficient for each SIC code. The coefficient is a metric that describes how well clustered a system is. That is how close points in a cluster are to other points in that cluster, as opposed to being close to points belonging to other clusters. In our case, we do this based on the company descriptions to understand whether companies labelled with a SIC code are clustered together with other companies with semantically similar descriptions, or whether they are dispersed in poorly defined clusters. 

To generate this value, we must first project the companies into some quantitative space that represents the semantics of their descriptions. We use another pre-trained DistilBERT based transformer model for this task. We choose a version of this model that has been trained on semantic similarity tasks and without any further fine-tuning we process the company descriptions to produce dense 768 dimensional vector representations of the texts.

With each of the companies now occupying some point in semantic space, we calculate the mean sample silhouette coefficient for each SIC code. The sample silhouette score for a single company, $s(x)$, is defined as

$$ s(x) = \frac{b(x) - a(x)}{\max\{a(x),b(x)\}} $$

where $a(x)$ is the distance of a company description from all other companies with the same 4-digit SIC label and $b(x)$ is its distance from all other companies. We use the cosine distance as our distance metric. We then take the mean of the sample silhouette coefficient for all companies that are labelled with a 4-digit SIC code in the test set.

By comparing these two metrics we can see the degree to which companies within a code having dispersed descriptions determines the range of predicted SIC codes. Indeed, [#fig:silhouette_shannon] shows that a SIC code with a higher silhouette coefficient tends to lead to less diverse predictions. This makes intuitive sense as SIC codes whose company descriptions are less tightly clustered will necessarily have some overlap with other codes. This will result in a challenging situation for the classifier when it comes to learning and predicting for company descriptions that are in dispersed clusters or are on the periphery of their group. Almost all of the SIC codes have a silhouette score below zero, highlighting the degree to which this is the case. This means that in the majority of codes, company descriptions can be interpreted as more similar to companies in other industries than their own. It is also notable that there are some SIC codes which have a range of Silhouette scores but a Shannon index of zero, indicating that all companies in the test set with this code have been predicted into the same class. 

![Comparing the silhouette score of company description embeddings and Shannon index of predictions for each 4-digit SIC code shows that SIC codes with less well defined semantic space are more likely generate a diverse array of predictions.](tables_figures/pred_shannon_vs_test_silhouette_scatter.png){#fig:silhouette_shannon}

However, a low Shannon index, driven by a high silhouette score does not necessarily mean that companies in that sector are well classified. 
Sectors with more dispersed company descriptions are resolved into more appropriate sectors





As the F1 score is the harmonic mean of precision and recall, it provides a view of model performance that attempts to balance these two measures of accuracy. This obscurs the fact that for this task, there are codes which do not necessarily share the same level of performance across both metrics. 

## 2.3 Discussion

## 2.4 Conclusion
















3512 Transmission of electricity
Sample(index=135032, text='O’Hanlon Electrical Ltd (OEL) was established in 1976 by founder Brendan O’Hanlon. OEL has over 39 years of experience offering Electrical Building Services: Electrical installations, Electrical maintenance and Electrical Testing to Public, Commercial, Industrial and Private sector clients across the UK and Ireland. We are a leading electrical contractor, offering an industry recognised reputation for electrical contracting providing a professional service, delivering installations on time, to customer’s exact requirements and at a competitive price, making us a popular choice for our clients. Our offices are located just off Junction 14 on the M1 Motorway.', label=480) 

4321 Electrical installation
Sample(index=586584, text='Alan Benfield Ltd since its creation 40\xa0years ago in 1976 has established its credentials as one of the Midlands leading Electrical Contractors. Our extensive experience in a vast array of electrical services from Domestic to Industrial, from Data and Voice Installations to LV Switchgear and Distribution. Managing Director Paul Waldron and his team offer the very best in expertise and knowledge to provide the most cost effective solution to your electrical needs using the latest technology. Using our state of the art CAD facility we can provide Electrical Design and Lighting Design along with Testing and Inspection certification. As a CHAS compliant Company and Construction Line member we provide all necessary Health and Safety documentation for all our projects no matter how small. We constantly update our employee training programme to ensure high compliance and low risk. With an extensive local client database Alan Benfield Ltd consistently hits the mark in aiming to provide the very best in Electrical Contracting and Maintenance.', label=21)