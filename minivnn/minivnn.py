import numpy as np
from typing import List, Tuple


def normalize(embedding: np.ndarray) -> np.ndarray:
    return embedding / np.linalg.norm(embedding)


class Index:
    def __init__(self, dim: int, metric: str = "dot") -> None:
        self.dim = dim
        self.metric = metric
        self.embeddings = np.empty((0, dim))
        self.index_map: List[int] = []

    def add(self, index: int, embedding: np.ndarray) -> None:
        if index in self.index_map:
            raise ValueError(f"Index {index} already exists.")

        embedding = embedding.reshape(1, -1)
        if embedding.shape[1] != self.dim:
            raise ValueError(f"Embedding has invalid dimension: {embedding.shape}. Expected dimension: {self.dim}.")
        if self.metric == "cosine":
            embedding = normalize(embedding)

        self.embeddings = np.append(self.embeddings, embedding.reshape(1, -1), axis=0)
        self.index_map.append(index)

    def delete(self, index: int) -> None:
        if index not in self.index_map:
            raise ValueError(f"Index {index} not found.")
        row_to_delete = self.index_map.index(index)
        self.embeddings = np.delete(self.embeddings, row_to_delete, axis=0)
        del self.index_map[row_to_delete]

    def save(self, filepath: str) -> None:
        np.savez_compressed(filepath, embeddings=self.embeddings, index_map=np.array(self.index_map))

    def load(self, filepath: str) -> None:
        with np.load(filepath) as data:
            self.embeddings = data["embeddings"]
            self.index_map = data["index_map"].tolist()

    def query(self, query_embedding: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        if self.metric == "dot":
            similarities = np.dot(self.embeddings, query_embedding)
        elif self.metric == "cosine":
            normalized_query = normalize(query_embedding)
            similarities = np.dot(self.embeddings, normalized_query)
        else:
            raise ValueError(f"Invalid metric: {self.metric}. Supported metrics are 'dot' and 'cosine'.")

        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices_sorted = top_k_indices[np.argsort(-similarities[top_k_indices])]
        top_k_values = similarities[top_k_indices_sorted]

        return [(self.index_map[i], similarity) for i, similarity in zip(top_k_indices_sorted, top_k_values)]
