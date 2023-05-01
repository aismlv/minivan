import numpy as np

from minivan.metrics import (
    cosine_metric,
    dot_product_metric,
    euclidean_metric,
    normalize,
)


def test_normalize():
    embedding = np.array([0.1, 0.2, 0.3])
    normalized_embedding = normalize(embedding)
    assert np.allclose(normalized_embedding, embedding / np.linalg.norm(embedding))
    assert np.allclose(np.power(normalized_embedding, 2).sum(), 1)

    embedding = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    normalized_embedding = normalize(embedding)
    assert np.allclose(normalized_embedding, embedding / np.linalg.norm(embedding, axis=1, keepdims=True))
    assert np.allclose(np.power(normalized_embedding, 2).sum(axis=1), 1)


def test_dot_product_metric():
    query_embedding = np.array([0.1, 0.2, 0.3])
    embeddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])

    assert np.allclose(dot_product_metric(query_embedding, embeddings), np.array([0.14, 0.20]))


def test_cosine_metric():
    query_embedding = np.array([0.1, 0.2, 0.3])
    embeddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])

    assert np.allclose(cosine_metric(query_embedding, normalize(embeddings)), np.array([1, 0.992583]))


def test_euclidean_metric():
    query_embedding = np.array([0.1, 0.2, 0.3])
    embeddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])

    assert np.allclose(euclidean_metric(query_embedding, embeddings), np.array([-0, -0.173205]))
