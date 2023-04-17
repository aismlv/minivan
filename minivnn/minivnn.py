import numpy as np


class Index:
    def __init__(self, dim: int, metric="dot"):
        self.metric = metric
        self.embeddings = np.empty((0, dim))
        self.index_map = []

    def add(self, index, embedding):
        if index in self.index_map:
            raise ValueError(f"Index {index} already exists")
        self.embeddings = np.append(self.embeddings, embedding.reshape(1, -1), axis=0)
        self.index_map.append(index)

    def delete(self, index):
        if index not in self.index_map:
            raise ValueError(f"Index {index} not found")
        row_to_delete = self.index_map.index(index)
        self.embeddings = np.delete(self.embeddings, row_to_delete, axis=0)
        del self.index_map[row_to_delete]

    def save(self, filepath):
        np.savez_compressed(filepath, embeddings=self.embeddings, index_map=np.array(self.index_map))

    def load(self, filepath):
        with np.load(filepath) as data:
            self.embeddings = data["embeddings"]
            self.index_map = data["index_map"].tolist()

    def query(self, query_embedding, k=1):
        if self.metric == "dot":
            similarities = np.dot(self.embeddings, query_embedding)
        elif self.metric == "cosine":
            normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            similarities = np.dot(normalized_embeddings, normalized_query)
        else:
            raise ValueError(f"Invalid metric: {self.metric}. Supported metrics are 'dot' and 'cosine'.")

        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices_sorted = top_k_indices[np.argsort(-similarities[top_k_indices])]
        top_k_values = similarities[top_k_indices_sorted]

        return [(index, similarity) for index, similarity in zip(top_k_indices_sorted, top_k_values)]
