# Solution building

## Metric

Text detoxification is text to text task, for which is quite hard to use any predefined metric. We can use BLEU metric but it will not work on data without given translation and it does not care about semantic meaning. It is important because there is no optimal way of detoxification and multiple variant are acceptable. Thus, metric was proposed:
$$
    \text{score} = \frac{1}{N} \sum_{i=1}^{N} (P_t(r_i) - P_t(t_i)) \cdot \text{S}(r_i, t_i)

$$
Where 
- $P_t$ probability of text being toxic. Range $[0, 1]$
- $r$ testing/reference dataset, where $r_i$ is sentence, which need to be detoxified.
- $t$ translated dataset, where $t_i$ is a detoxified version of $r_i$. 
- $S(r_i, t_i)$ semantic similarity between two texts. Range $[0, 1]$

To compute this metric $P_t$ will be trained using given dataset and $S(r_i, t_i)$ will be pretrained [model](https://www.sbert.net/).

### Hypothesis 1: `Logistic regresion with Bag of Words (BoW)`
In the given paper for classifying toxicity simple logistic regression is used. So it was decided to use it like baseline for classification.

Tokenization was done in following manner:
- replace all punctuation and garbage with space using this regexp: ```r"[();.\/\\$%,!&?*'\"\’—–~…\”-]"```
- remove all repeating spaces
- split using spaces
- remove NLTK stopwords
- Stem tokens using NLTK

BoW was using 2000 most frequent tokens.

Labels were used as in dataset.

#### Result

Result was on test dataset using Cross Entropy Loss is
`0.462955`

### Hypothesis 2: `Logistic regression with weighted BoW`

Let's use Weighted BoW where will we use not most frequent words but words which are presented in most toxic sentences on average. Words with less than 100 occurences are ignored.

Everything else is the same.

#### Result
Result was on test dataset using Cross Entropy Loss is
`0.461165`


### Hypothesis 3: `Logistic regression with weighted BoW and hard labels`

Let's use hard labels instead of soft labels by using `0.5` threshold.

#### Result
Result was on test dataset using Cross Entropy Loss is
`0.453793`

### Hypothesis 4: `DistilBERT`

Logistic regression based approaches did not gave good results which you can see in [notebook](../notebooks/ToxicityClassificationExploration.ipynb). To solve this problem we can use LLM such as DistilBERT. 

Model was trained on following [parameters](../configs/sentence_toxic_cls/distilbert.yaml), most important:
- Maximum token length is 32 then truncated
- Adam with weight decay with lr = `0.00005`
- Batch size `512`

#### Result
Result was on test dataset using Cross Entropy Loss is
`0.05`. Best result obtained

## Detoxification

For detoxification itself T5 based model was seen as best variant.

### Hypothesis 1: `T5-small`

Intention was to use as little model as possible. However, this model showed strange results on manual evaluation. It includes: just repeating text, translating to german, tranlsating to french.

Parameters were used:
- AdamW, lr 0.0001
- Maximum token length 32

### Hypothesis 2: `T5-base`:

This model produces best results so far.
Parameters were used:
- AdamW, lr 0.001

### Hypothesis 3:

Idea was to use T5-large or flan. However, they were to big to train.
