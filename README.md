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

- [CS224N: Natural Language Processing with Deep Learning (lectures 1-14)](http://web.stanford.edu/class/cs224n/)
- [Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space"](https://arxiv.org/abs/1301.3781)
- [Bojanowski, Piotr, et al. "Enriching word vectors with subword information" ](https://arxiv.org/abs/1607.04606)
- [Reimers, Nils, and Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"](https://arxiv.org/pdf/1908.10084.pdf)

  - [Sentrance Transformers on Hugging Face](https://huggingface.co/sentence-transformers)
  - Their solo [website](https://www.sbert.net/)
    which claims

    > You can use this framework to compute sentence / text embeddings for more than 100 languages.

    and contains a list of [pretrained models](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/)

- Ghasemi, Negin, and Saeedeh Momtazi. "Neural text similarity of user reviews for improving collaborative filtering recommender systems"
