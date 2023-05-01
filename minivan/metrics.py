import numpy as np

DOT_PRODUCT = "dot_product"
COSINE = "cosine"
EUCLIDEAN = "euclidean"


def normalize(embedding: np.ndarray) -> np.ndarray:
    return embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)


def dot_product_metric(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ query_embedding


def cosine_metric(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    # assumes that embeddings are already normalized
    return embeddings @ normalize(query_embedding)


def euclidean_metric(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    return -np.linalg.norm(embeddings - query_embedding, axis=1)


def get_metric(metric: str):
    if metric == "dot_product":
        return dot_product_metric
    elif metric == "cosine":
        return cosine_metric
    elif metric == "euclidean":
        return euclidean_metric
    else:
        raise ValueError(f"Invalid metric: {metric}. Supported metrics are: {DOT_PRODUCT}, {COSINE}, {EUCLIDEAN}.")
