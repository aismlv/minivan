import numpy as np
import pytest

from minivan import Index
from minivan.index import normalize


def test_normalize():
    embedding = np.array([0.1, 0.2, 0.3])
    normalized_embedding = normalize(embedding)
    assert np.allclose(normalized_embedding, embedding / np.linalg.norm(embedding))
    assert np.allclose(np.power(normalized_embedding, 2).sum(), 1)

    embedding = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    normalized_embedding = normalize(embedding)
    assert np.allclose(normalized_embedding, embedding / np.linalg.norm(embedding, axis=1, keepdims=True))
    assert np.allclose(np.power(normalized_embedding, 2).sum(axis=1), 1)


def test_add_items_list():
    index = Index(3, metric="dot_product")
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])

    index.add_items([1, 2], [embedding1, embedding2])
    assert index.embeddings.shape == (2, 3)
    assert len(index.index_map) == 2
    assert index.index_map[0] == 1
    assert np.allclose(index.embeddings[0], embedding1)

    with pytest.raises(KeyError):
        index.add_items([1], [embedding1])


def test_add_items_2d_array():
    index = Index(3, metric="dot_product")
    embeddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])

    index.add_items([1, 2], embeddings)
    assert index.embeddings.shape == (2, 3)
    assert len(index.index_map) == 2
    assert index.index_map[0] == 1
    assert np.allclose(index.embeddings[0], embeddings[0])

    with pytest.raises(ValueError):
        index.add_items([3], embeddings)


def test_add_items_cosine():
    index = Index(3, metric="cosine")
    embedding = np.array([0.1, 0.2, 0.3])

    index.add_items([1], [embedding])
    assert index.embeddings.shape == (1, 3)
    assert len(index.index_map) == 1
    assert index.index_map[0] == 1
    assert np.allclose(index.embeddings[0], embedding / np.linalg.norm(embedding))

    with pytest.raises(KeyError):
        index.add_items([1], [embedding])


def test_delete_items():
    index = Index(3)
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])

    index.add_items([1, 2], [embedding1, embedding2])

    index.delete_items([1])
    assert index.embeddings.shape == (1, 3)
    assert np.allclose(index.embeddings[0], embedding2)
    assert 1 not in index.index_map
    assert index.index_map[0] == 2

    with pytest.raises(KeyError):
        index.delete_items([1])


def test_save_load(tmp_path):
    index = Index(3, metric="dot_product")
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])

    index.add_items([1, 2], [embedding1, embedding2])

    filepath = tmp_path / "index_data.npz"
    index.save(filepath)

    loaded_index = Index(3, metric="dot_product")
    loaded_index.load(filepath)

    assert np.allclose(loaded_index.embeddings, index.embeddings)
    assert loaded_index.index_map == index.index_map

    loaded_index = Index(3, metric="cosine")
    with pytest.raises(ValueError):
        loaded_index.load(filepath)

    loaded_index = Index(2, metric="dot_product")
    with pytest.raises(ValueError):
        loaded_index.load(filepath)

    loaded_index = Index(3, metric="dot_product", dtype="float64")
    with pytest.raises(ValueError):
        loaded_index.load(filepath)


def test_load_from_file(tmp_path):
    index = Index(3, metric="dot_product")
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])

    index.add_items([1, 2], [embedding1, embedding2])

    filepath = tmp_path / "index_data.npz"
    index.save(filepath)

    loaded_index = Index.from_file(filepath)
    assert np.allclose(loaded_index.embeddings, index.embeddings)
    assert loaded_index.index_map == index.index_map


def test_query_dot_product():
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])
    embedding3 = np.array([0.3, 0.4, 0.5])
    query_embedding = np.array([0.15, 0.25, 0.35])

    index = Index(3, metric="dot_product")
    index.add_items([2, 3, 4], [embedding1, embedding2, embedding3])
    result = index.query(query_embedding, k=2)

    assert len(result) == 2
    assert result[0][0] == 4
    assert result[1][0] == 3


def test_query_cosine():
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])
    embedding3 = np.array([0.3, 0.4, 0.5])
    query_embedding = np.array([0.15, 0.25, 0.35])

    index = Index(3, metric="cosine")
    index.add_items([2, 3, 4], [embedding1, embedding2, embedding3])
    result = index.query(query_embedding, k=2)

    assert len(result) == 2
    assert result[0][0] == 3
    assert result[1][0] == 2


def test_len():
    index = Index(3)
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])

    assert len(index) == 0

    index.add_items([1, 2], [embedding1, embedding2])
    assert len(index) == 2

    index.delete_items([1])
    assert len(index) == 1


def test_get_item():
    index = Index(3)
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])

    index.add_items([1, 2], [embedding1, embedding2])
    assert np.allclose(index.get_items([1]), embedding1)
    assert np.allclose(index.get_items([1, 2]), np.vstack([embedding1, embedding2]))

    with pytest.raises(KeyError):
        index.get_items([3])
