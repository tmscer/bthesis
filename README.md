[![Build paper](https://github.com/tmscer/text-embeddings-for-recommenders/actions/workflows/paper.yml/badge.svg)](https://github.com/tmscer/text-embeddings-for-recommenders/actions/workflows/paper.yml)
[![Docker image with TexLive and Pandoc](https://github.com/tmscer/text-embeddings-for-recommenders/actions/workflows/paper-docker.yml/badge.svg)](https://github.com/tmscer/text-embeddings-for-recommenders/actions/workflows/paper-docker.yml)

# Text Embeddings for Recommender Systems

This is my bachelor thesis at CTU FEE supervised by Ing. Jan Drchal, Ph.D. ([Google Scholar](https://scholar.google.cz/citations?user=JL9IGwcAAAAJ), [CTU website](https://cs.felk.cvut.cz/en/people/drchajan))

### Description

The task is to compare various embeddings and their usability in recommender systems.

### Requirements

1. Research state-of-the-art in text embeddings. Explore both contextual (e.g., BERT) and non-contextual (e.g., Word2Vec) approaches.
2. Select existing dataset and build a recommender system.
3. Select an appropriate evaluation methodology.
4. Compare embeddings with TF-IDF and/or similar baseline.

# Report

See it [here](https://f35ba6fb-bthe.s3.eu-west-1.amazonaws.com/paper.pdf). It's generated automatically on every relevant commit.

# Resources

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

