# minivnn

![Tests](https://github.com/aismlv/minivnn/actions/workflows/test_and_lint.yml/badge.svg)
[![codecov](https://codecov.io/gh/aismlv/minivnn/branch/main/graph/badge.svg?token=5J503UR8O7)](https://codecov.io/gh/aismlv/minivnn)
[![PyPI version](https://badge.fury.io/py/minivnn.svg)](https://pypi.org/project/minivnn/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`minivnn` (pronounced "minivan" 🚐) is an exact nearest neighbor search Python library for those times when "approximate" just won't cut it (or is simply overkill).

## Installation

Install `minivnn` using `pip`:

```bash
pip install minivnn
```

## Usage
Here's an example of how to use minivnn:

```python
from minivnn import Index
import numpy as np

# Create an index with 128-dimensional embeddings and cosine similarity metric
index = Index(dim=128, metric="cosine")

# Add embeddings to the index
embedding1 = np.random.rand(128)
embedding2 = np.random.rand(128)
embedding3 = np.random.rand(128)

index.add_items([1, 2, 3], [embedding1, embedding2, embedding3])
index.add_items([4, 5, 6], np.random.rand(3, 128))

# Delete embeddings from the index
index.delete_items([3])

# Query the index for the nearest neighbor of a given embedding
query_embedding = np.random.rand(128)
result = index.query(query_embedding, k=1)

print(result)  # Returns [(index, similarity)] of the nearest neighbor
```

Cosine similarity and dot product are supported.
