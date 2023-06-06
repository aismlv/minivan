# minivan

![Tests](https://github.com/aismlv/minivan/actions/workflows/test_and_lint.yml/badge.svg)
[![codecov](https://codecov.io/gh/aismlv/minivan/branch/main/graph/badge.svg?token=5J503UR8O7)](https://codecov.io/gh/aismlv/minivan)
[![PyPI version](https://badge.fury.io/py/minivan-tools.svg?)](https://pypi.org/project/minivan-tools/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`minivan` is an exact nearest neighbor search Python library for those times when "approximate" just won't cut it (or is simply an overkill).

## Installation

Install `minivan` using `pip`:

```bash
pip install minivan-tools
```

## Usage
Create new index:
```python
from minivan import Index
import numpy as np

# Create an index with 128-dimensional embeddings and dot product metric
index = Index(dim=128, metric="dot_product")

# Add embeddings to the index
embeddings = [np.random.rand(128) for _ in range(3)]
index.add_items([1, 2, 3], embeddings)

# Delete embeddings from the index
index.delete_items([3])
```

Query the index for the nearest neighbor:
```python
query_embedding = np.random.rand(128)
result = index.query(query_embedding, k=1)

print(result)  # Returns [(index, similarity)] of the nearest neighbor
```

Save the index for future use:
```python
# Save to disk
index.save(filepath)

# Load from a saved file
new_index = Index.from_file(filepath)
```

## `np.matmul` vs ANN

In most cases you would want to go with an ANN solution such as [hnswlib](https://github.com/nmslib/hnswlib), but due to numpy's use of BLAS and other optimisations, brute-force search is performant enough for a large set of applications (see a [quick benchmark](https://github.com/aismlv/minivan/blob/main/experiments/benchmark/README.md) for an illustration.) There are a bunch of cases when you might want to start with a simpler approach:

- Your document set is not in the multiple millions
- You're in the experimentation phase and want to iterate on the index rapidly with fast build times
- Your application requires the best possible accuracy
- You want to avoid the need to finetune hyperparameters (which can affect [performance and latency](https://github.com/erikbern/ann-benchmarks) quite a lot)
