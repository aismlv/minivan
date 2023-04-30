# minivn

![Tests](https://github.com/aismlv/minivn/actions/workflows/test_and_lint.yml/badge.svg)
[![codecov](https://codecov.io/gh/aismlv/minivn/branch/main/graph/badge.svg?token=5J503UR8O7)](https://codecov.io/gh/aismlv/minivn)
[![PyPI version](https://badge.fury.io/py/minivn.svg)](https://pypi.org/project/minivn/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`minivn` (pronounced "minivan" 🚐) is an exact nearest neighbor search Python library for those times when "approximate" just won't cut it (or is simply overkill).

## Installation

Install `minivn` using `pip`:

```bash
pip install minivn
```

## Usage
Here's an example of how to use minivn:

```python
from minivn import Index
import numpy as np

# Create an index with 128-dimensional embeddings and dot product metric. Cosine similarity is also supported
index = Index(dim=128, metric="dot_product")

# Add embeddings to the index
embedding1 = np.random.rand(128)
embedding2 = np.random.rand(128)
embedding3 = np.random.rand(128)

index.add_items([1, 2, 3], [embedding1, embedding2, embedding3])

# or
index.add_items([4, 5, 6], np.random.rand(3, 128))

# Delete embeddings from the index
index.delete_items([3])

# Query the index for the nearest neighbor of a given embedding
query_embedding = np.random.rand(128)
result = index.query(query_embedding, k=1)

print(result)  # Returns [(index, similarity)] of the nearest neighbor

# Save and load the index
index.save(filepath)

new_index = Index(dim=128, metric="dot_product")
new_index.load(filepath)
```

## Comparison with Approximate Nearest Neighbor Search
Based on a [quick benchmark](https://github.com/aismlv/minivn/blob/main/benchmark/README.md), you might not require an ANN and go with a simpler approach if any of the below apply:

- Your document set isn't in the multiple millions and you don't need ultra-high latency requirements (to, for example, accommodate a heavy reranker)
- You're in the experimentation phase and want to iterate quickly on the index
- Your application requires the best precision and recall
- You don't want to fine-tune any hyperparameters (which can affect [latency/accuracy trade-off](https://github.com/erikbern/ann-benchmarks) quite a lot)
