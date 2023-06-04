# Project Report - Text Embeddings for Recommender Systems

Author: Tomáš Černý

Supervisor: Ing. Jan Drchal, Ph.D.

## Paper Notes

### [CS224N: Natural Language Processing with Deep Learning (lectures 1-14)](http://web.stanford.edu/class/cs224n/)

Gives good introduction to NLP and neural networks. This run is from 2021 so OpenAI's advances
aren't mentioned but that isn't that relevant to this project which is more about applying embeddings.

### [Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space"](https://arxiv.org/abs/1301.3781)

### [Bojanowski, Piotr, et al. "Enriching word vectors with subword information" ](https://arxiv.org/abs/1607.04606)

Word embeddings are composed of n-grams, e.g. word "where" as a 3-gram would be "&lt;wh",
"whe", "her" (not the same as "&lt;her&gt;"), "ere", "re&gt;" and special "&lt;where&gt;.
This is useful for languges with rich morphology, e.g. Czech and other slavic languages.
They use the skip-gram model from Word2Vec with this subword information and outperform the
original on several benchmarks or come very close to it.

### [Reimers, Nils, and Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"](https://arxiv.org/pdf/1908.10084.pdf)

- [Sentence Transformers on Hugging Face](https://huggingface.co/sentence-transformers)
- Their solo [website](https://www.sbert.net/)
  which claims

  > You can use this framework to compute sentence / text embeddings for more than 100 languages.

  and contains a list of [pretrained models](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/)

### Ghasemi, Negin, and Saeedeh Momtazi. "Neural text similarity of user reviews for improving collaborative filtering recommender systems"

Their proposed approach outperforms baseline model that uses only product ratings and
a model called [DeepCoNN by Zheng et al. (2017)](https://arxiv.org/pdf/1701.04783.pdf)
by utilizing ratings and text similarity of reviews.

They claim their method is efficient on CPU whereas other complex neural models are
computationally demanding.

The approach to ratings could be used for article likes/dislikes (or emojis for more
granular feedback) as well as to how many % of an article a user has read.

Text similarity could be used for discussion comments, although those can contain opinions
on article, topic and author. It would be ideal to extract some form of opinion on all three. In contrast, product reviews are more objective and have a narrower focus.

Nevertheless, we can try different text similarity methods - we can try different embeddings.
E.g. the most complex model they used was an autoencoder LSTM. There are transformer-based
models that could provide better embeddings.

Might use their evaluation metric Root Mean Squared Error (RMSE).

### [Kocián, Náplava, Štancl, and Kadlec. "Siamese BERT-based Model for Web Search Relevance Ranking Evaluated on a New Czech Dataset"](https://arxiv.org/pdf/2112.01810.pdf)

Their goal was to have a small and fast model that performs well, that's why they used the Electra-small architecture.

The model is pretrained using by random token change which the model is supposed to detect. They then fine tune the model
for several tasks such as web search relevance ranking (in the paper) and on GitHub they mention it being used for clickbait
title detection, query typo correction and sentiment analysis.

In web search, they use it for stage 2 ranking on around 20 thousand documents given a query for ranking. In contrast, the previous
paper uses the models for embeddings only and then uses some distance metric for ranking. For use in web search, they created
two finetuned variants, one is a query-document which isn't practical because it needs to go over all documents for a query, the other
is a siamese model that produces embeddings which can be fed to cosine distance, euclid distance, pooling, etc. The output is
then fed to a feed-forward network to produce the final score. The more computationally expensive variant query-doc was used
as a teacher for the siamese model.

One downside of this model is that it needs to be finetuned and their new DaReCzech dataset for search ranking is provided per request
for non-commercial uses. This model cannot be therefore used for a content-based recommender system. They however, mentioned other
models which are either multilingual or specificly trained for Czech language I should check out: mBERT and RobeCzech.

The model is available at [HuggingFace seznam/small-e-czech](https://huggingface.co/Seznam/small-e-czech).

### [Takács, and Tikk. "Alternating least squares for personalized ranking"](https://www.researchgate.net/publication/254464370_Alternating_least_squares_for_personalized_ranking)

This paper mainly demonstrates how an objective function and its equivalent forms can impact training speed.
It also handles implicit feedback (e.g. a user read an article but didn't like it/rate it/comment on it) which
is a useful property as reading an article (and what percentage the user read) is an implicit feature.

### [Maciej Kula. "Metadata Embeddings for User and Item Cold-start Recommendations"](https://arxiv.org/pdf/1507.08439.pdf)

A hybrid model (called LightFM) that should work well when content features are missing but there are many interactions for collaborative filtering and vice versa.
Online learning and addition of features to items is possible. Learning custom chosen features (human and from other systems) and embeddings for
items and users at the same time. They used AUC metric for evaluation on the MovieLens 10M and CrossValidated datasets.

This means several things:

1. We can give items fixed textual embeddings.
2. We can give items latent features such as which author wrote it and consider authors as individual features with unknown values.
3. LightFM used tags as features and later was able to find similar tags. This allows us to learn embeddings for any categorization of items.
4. We can also dynamically add types of categorization (tags, etc.).
5. We can compute other items's features using specialized models (such as clickbaitiness and sentiment).

This simplifies the problem by having unified model that can leverage both content-based and collaborative filtering.
Originally, my mental model was that there would be independent collaborative model evaluated e.g. on MovieLens dataset and content-based
filtering would be done using distance of article embeddings.

However, there is [critique of LightFM's hybrid approach](https://amanda-shu.medium.com/lightfm-performance-7515e57f5cfe):

> In this article, we have evaluated the LightFM code package and compared the performance of its pure and hybrid models along with baseline algorithms on the MovieLens dataset.
> Although Dacrema’s work indicates that many algorithms that have been published in the field of recommendation systems have struggled to beat out baselines, we find that
> LightFM’s pure collaborative filtering model outperformed all baselines for the precision and recall metrics for a majority of the cutoffs. However, we find the LightFM’s
> hybrid model with item features did not perform as well as a few baseline algorithms, such as ItemKNN. The LightFM-hybrid also performed worse than the pure LightFM model,
> which falls in line with work by other data scientists who use the LightFM package in practice. While the pure LightFM model outperformed the baselines as expected,
> the inability of the LightFM’s hybrid model to beat some of these baselines show that even with commonly used code packages, there may be still issues in their ability to
> deliver basic expectations.

But they also wrote:

> Our results show that after optimization of parameters for the baseline algorithms, the pure LightFM model outperforms all of the
> baselines for both precision and recall at all cutoffs, with the exception of precision@5, where ItemKNN is the best. This result
> is in agreement with our hypothesis, and it indicates that LightFM’s pure collaborative model is a good choice for data scientists
> to use in practice.

Possible explanation: bad item features, it's evaluated on one dataset and it is MovieLens 100k but the original LightFM paper used the 10M variant and was also evaluated
on the CrossValidated dataset where they used user's about text as a feature using a bag-of-words representation.

I should explore [Dacrema et al. "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches"](https://arxiv.org/pdf/1907.06902.pdf) to get more context into
baseline recommender algorithms that the critique blogpost mentions.

### [Dacrema et al. "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches"](https://arxiv.org/pdf/1907.06902.pdf)

They demonstrate tuned simple approaches often outperform complex neural models. They state several factors contribute to this phenomenon:

1. Weak baselines are used or they aren't tuned.
2. Weak methods are established as baselines as it isn't clear what the state-of-the-art method is.
3. It's hard to compare and reproduce results as datasets and evaluation metrics are different and often not well justified.

List of used simple methods:

- Non-personalized

  - TopPopular: Recommends the most popular items to everyone

- Nearest-Neighbor

  - UserKNN: User-based k-nearest neighbors
  - ItemKNN: Item-based k-nearest neighbors

- Graph-based

- P3alpha: A graph-based method based on random walks
- RP3beta: An extension of P3alpha

- Content-Based and Hybrid

  - ItemKNN-CBF: ItemKNN with content-based similarity
  - ItemKNN-CFCBF: A simple item-based hybrid CBF/CF approach
  - UserKNN-CBF: UserKNN with content-based similarity
  - UserKNN-CFCBF: A simple user-based hybrid CBF/CF approach

- And some other non-neural machine learning approaches

### [SurpriseLib](https://surprise.readthedocs.io/en/stable)

A library for baseline recommender systems. Previously implemented UserKNN model was very similar
to what I've assumed was used in Dacrema et al. 2019 (I was missing a normalization factor).


## Experiments

Dataset CiteULike-a was chosen for these experiments because the recommended items have
textual content which can be used for embeddings. Even though the dataset is relatively small,
the embeddings for all items still take up ~1GB.

We calculate precision, recall, their harmonic mean (F1 score) and root mean squared error (RMSE)
for each approach.

### Implemented approaches

All approaches are some variant of collaborative UserKNN. ItemKNN approaches weren't explored although a straightforward
embedding search using cosine similarity given a query yielded superficially which is what the embeddings are supposed
to provide. We used the `all-mpnet-base-v2` model from [sentence-transformers](https://www.sbert.net/).

In all cases, given a user, we take the user-item matrix and find K most similar users
by cosine similarity (except the given user). We then infer the rating by taking K nearest
users and weigh them by several methods:

- cosine similarity of user vectors from user-item matrix
- cosine similarity of title embeddins
- cosine similarity of abstract embeddins
- cosine similarity of title+abstract (labelled "text" in code) embeddings

There are also embedding variants where we use mean and max similarity of other users
to the given user. The max variant is inspired by Ghasemi et al. 2021 where they
used max-similarity of product reviews of two users.

Pseudocode:

```python
def calculate_scores(user_id, k):
  user_similarities = cosine_similarity(user_item_matrix[user_id], user_item_matrix)
  similar_users = np.argsort(user_similarities)[::-1][:k + 1]
  item_scores = np.zeros(num_items)

  for similar_users in similar_users:
    if user_id == similar_user:
      continue

    # `calculate_user_similarity` is one of the methods described above
    user_similarity = calculate_user_similarity(user_id, similar_user)
    item_scores += user_similarity * user_item_matrix[similar_user]

  return item_scores
```

We also added a random score method as a sanity check.

Also, we tried to combine two of the aforementioned methods by taking a weighted average.
Namely user similarity and title+abstract similarity of users' items and chose the one
with the highest F1 score.

```python
for w in np.arange(0.65, 0.75, 0.025):
	weights = [w, 1 - w]
	result = evaluate(10, ["user-similarity", "text-similarity-mean"], weights)
	f1 = f1_score(result["precision"], result["recall"])
	print(w, f1, result)
```

## Results

Floats are rounded to 4 decimal places. Omitting title and abstract embeddings and max pooling as it performed
worse than mean pooling.

| (forget rate, k, method)                            |     f1 |   hitrate |   precision |   recall |
|:----------------------------------------------------|-------:|----------:|------------:|---------:|
| (0.97, 5, 'random')                                 | 0.0003 |    0.0065 |      0.0013 |   0.0002 |
| (0.97, 5, 'text-similarity-mean')                   | 0.0846 |    0.7085 |      0.2521 |   0.0508 |
| (0.97, 5, 'user-similarity')                        | 0.085  |    0.7073 |      0.2551 |   0.051  |
| (0.97, 5, 'user-similarity, text-similarity-mean')  | 0.0879 |    0.7235 |      0.2647 |   0.0527 |
| (0.97, 10, 'random')                                | 0.0009 |    0.0189 |      0.0019 |   0.0006 |
| (0.97, 10, 'text-similarity-mean')                  | 0.1158 |    0.8438 |      0.2052 |   0.0807 |
| (0.97, 10, 'user-similarity')                       | 0.1151 |    0.8435 |      0.2051 |   0.08   |
| (0.97, 10, 'user-similarity, text-similarity-mean') | 0.1202 |    0.8519 |      0.2153 |   0.0833 |
| (0.97, 25, 'random')                                | 0.0016 |    0.047  |      0.002  |   0.0013 |
| (0.97, 25, 'text-similarity-mean')                  | 0.1474 |    0.955  |      0.1527 |   0.1425 |
| (0.97, 25, 'user-similarity')                       | 0.1443 |    0.956  |      0.1491 |   0.1398 |
| (0.97, 25, 'user-similarity, text-similarity-mean') | 0.1511 |    0.9586 |      0.1573 |   0.1453 |
| (0.97, 50, 'random')                                | 0.0024 |    0.096  |      0.0021 |   0.0029 |
| (0.97, 50, 'text-similarity-mean')                  | 0.1491 |    0.9885 |      0.117  |   0.2053 |
| (0.97, 50, 'user-similarity')                       | 0.146  |    0.9897 |      0.1142 |   0.2025 |
| (0.97, 50, 'user-similarity, text-similarity-mean') | 0.1515 |    0.9901 |      0.1193 |   0.2077 |


As can be seen in the table, embeddings which are derived from content, are beneficial when there
is less information about users. This is a common observation in recommender systems (Kula. 2015 - LightFM).

It should be noted that we're always doing collaborative filtering, i.e. we're always using the user-item matrix
even when using text embeddings. All of the results (besides the random baseline) are relatively similar.

Result graphs for forget rate 0.97. Method "identity" is another sanity check, it's just returning the
user vector from trainig data.

![Results for forget_rate=0.97](./paper/results.png)

Comparing that to the results from Dacrema et al. 2019, we can see that our results are worse.
It isn't clear to thus how their methods were implemented even though their code is available
on [GitHub](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/tree/574c25520103106abea6f69f3905c98fd327fb28)
as is [ours](https://github.com/tmscer/text-embeddings-for-recommenders).

![Recall from Dacrema et al. 2019](./paper/decrema19-recall.png)
![Hitrate and NDCG from Dacrema et al. 2019](./paper/decrema19-hr-ndcg.png)

## Looking back

Variants of tried approaches and other approaches could be tried:

- optimizing for the shrinkage parameter in cosine similarity

  ```python  
  # `h` is called the shrinkage parameter
  def cosine_similarity(v1, v2, h=0.0):
      return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + h)
  ```

- using other suitable similarity measures (e.g. for the embeddings, only certain similarity measures are suitable according to their authors)
- using other recommender approaches, e.g. matrix factorization methods. As demonstrated in Dacrema et al. 2019, complex neural approaches
are likely to be outperformed by tuned simple methods.

## Mistakes

1. Better dataset hygiene could have been used. We've managed to extract train, validation and test splits from Dacrema et al. 2019 in the end
but only used the information how many user-item records were kept in trainig data (~2.7%, we kept 3% at random between hyperparam search and eval runs).
2. Not using random number generator seeds to make the experiments exactly reproducible.
3. Not evaluating purely content-based approaches since they have merit when there isn't a lot of collaborative data.
