[![Build paper](https://github.com/tmscer/text-embeddings-for-recommenders/actions/workflows/paper.yml/badge.svg?branch=master)](https://github.com/tmscer/text-embeddings-for-recommenders/actions/workflows/paper.yml)
[![Docker image with TexLive and Pandoc](https://github.com/tmscer/text-embeddings-for-recommenders/actions/workflows/paper-docker.yml/badge.svg?branch=master)](https://github.com/tmscer/text-embeddings-for-recommenders/actions/workflows/paper-docker.yml)

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

# Handwritten notes on everything

See them [here](https://f35ba6fb-bthe.s3.eu-west-1.amazonaws.com/notes.pdf). Heading on first page doesn't match,
the PDF also contains notes on recommender systems. Updates are uploaded sporadically.

# Cached computations

Some computations take a lot of time. That's why we provide cached embeddings, their similarity matrices and other files.
They can be downloaded from [here](https://f35ba6fb-bthe.s3.eu-west-1.amazonaws.com/cached-computations.tar),
the size is a few GB. Put them in `datasets/citeulike-a-cache/` and you're good to go.
