import logging
from typing import List, Tuple, Union

import numpy as np

from .metrics import COSINE, DOT_PRODUCT, get_metric, normalize


class Index:
    def __init__(self, dim: int, metric: str = DOT_PRODUCT, dtype: str = "float32") -> None:
        self.dim = dim
        self.metric = metric
        self.dtype = np.dtype(dtype)
        self.embeddings = np.empty((0, dim), dtype=dtype)
        self.index_map: List[int] = []

        self.calc_similarities = get_metric(metric)

    def add_items(self, indices: List[int], embeddings: Union[List[np.ndarray], np.ndarray]) -> None:
        if type(indices) != list:
            raise TypeError(f"Indices must be passed as a list. Got: {type(indices)}")

        for index in indices:
            if type(index) != int:
                raise TypeError(f"Index must be an integer. Got: {type(index)}")
            if index in self.index_map:
                raise KeyError(f"Index {index} already exists.")

        if isinstance(embeddings, list):
            embeddings = [embedding.reshape(1, -1) for embedding in embeddings]

            lenghts = set(embedding.shape[1] for embedding in embeddings)
            if len(lenghts) != 1:
                raise ValueError(f"Embeddings must have the same dimension. Got: {lenghts}")

            embeddings = np.vstack(embeddings)

        if not isinstance(embeddings, np.ndarray):
            raise TypeError(f"Embeddings must be a list or a numpy array. Got: {type(embeddings)}")

        if embeddings.shape != (len(indices), self.dim):
            raise ValueError(
                f"Embedding has invalid shape: {embeddings.shape}. Expected shape: {(len(indices), self.dim)}."
            )

        if self.metric == COSINE:
            embeddings = normalize(embeddings)

        self.embeddings = np.append(self.embeddings, embeddings.astype(self.dtype), axis=0)
        self.index_map.extend(indices)

    def delete_items(self, indices: List[int]) -> None:
        self._validate_index_exists(indices)

        rows_to_delete = [self.index_map.index(index) for index in indices]
        self.embeddings = np.delete(self.embeddings, rows_to_delete, axis=0)

        for index in rows_to_delete:
            del self.index_map[index]

    def save(self, filepath: str) -> None:
        np.savez_compressed(
            filepath,
            embeddings=self.embeddings,
            index_map=np.array(self.index_map),
            metric=self.metric,
            dim=self.dim,
            dtype=str(self.dtype),
        )

    def load(self, filepath: str) -> None:
        with np.load(filepath) as data:
            if data["metric"].item() != self.metric:
                raise ValueError(
                    f"Metric mismatch. Index metric: {self.metric}. Loaded index metric: {data['metric'].item()}."
                )
            if data["dim"].item() != self.dim:
                raise ValueError(
                    f"Dimension mismatch. Index dimension: {self.dim}. Loaded index dimension: {data['dim'].item()}."
                )
            if data["dtype"].item() != self.dtype:
                raise ValueError(
                    f"Dtype mismatch. Index dtype: {self.dtype}. Loaded index dtype: {data['dtype'].item()}."
                )
            self.embeddings = data["embeddings"]
            self.index_map = data["index_map"].tolist()

    @classmethod
    def from_file(cls, filepath: str) -> "Index":
        with np.load(filepath) as data:
            metric = data["metric"].item()
            dim = data["dim"].item()
            dtype = data["dtype"].item()
        index = cls(dim, metric, dtype)
        index.load(filepath)
        return index

    def query(self, query_embedding: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        if type(k) != int or k < 1:
            raise ValueError(f"k must be a positive integer: {k}")
        if k > len(self):
            raise ValueError(f"k cannot be greater than the number of items in the index: {len(self)}")
        if self.dim not in query_embedding.shape:
            raise ValueError(
                f"Query embedding has invalid dimension: {query_embedding.shape}. "
                f"Expected embedding dimension: {self.dim}."
            )

        query_embedding = query_embedding.astype(self.dtype)
        similarities = self.calc_similarities(query_embedding, self.embeddings)
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices_sorted = top_k_indices[np.argsort(-similarities[top_k_indices])]
        top_k_values = similarities[top_k_indices_sorted]

        return [(self.index_map[i], similarity) for i, similarity in zip(top_k_indices_sorted, top_k_values)]

    def __len__(self) -> int:
        return len(self.index_map)

    def __repr__(self) -> str:
        return f"Index(num_items={len(self)}, dim={self.dim}, metric={self.metric}, dtype={self.dtype})"

    def get_items(self, indices: List[int]) -> np.ndarray:
        self._validate_index_exists(indices)

        if self.metric == COSINE:
            logging.warning(
                (
                    f"The vector retrieved from the index using the {COSINE} metric is not equivalent "
                    "to the initially added embeddings as it has been normalized to have a unit length"
                )
            )

        return self.embeddings[[self.index_map.index(index) for index in indices]]

    def _validate_index_exists(self, indices: List[int]) -> None:
        for index in indices:
            if index not in self.index_map:
                raise KeyError(f"Index {index} not found.")
