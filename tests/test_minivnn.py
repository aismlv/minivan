import numpy as np
import pytest

from minivnn import Index
from minivnn.minivnn import normalize


def test_normalize():
    embedding = np.array([0.1, 0.2, 0.3])
    normalized_embedding = normalize(embedding)
    assert np.allclose(normalized_embedding, embedding / np.linalg.norm(embedding))


def test_add_dot():
    index = Index(3, metric="dot")
    embedding = np.array([0.1, 0.2, 0.3])

    index.add(1, embedding)
    assert index.embeddings.shape == (1, 3)
    assert len(index.index_map) == 1
    assert index.index_map[0] == 1
    assert np.allclose(index.embeddings[0], embedding)

    with pytest.raises(ValueError):
        index.add(1, embedding)


def test_add_cosine():
    index = Index(3, metric="cosine")
    embedding = np.array([0.1, 0.2, 0.3])

    index.add(1, embedding)
    assert index.embeddings.shape == (1, 3)
    assert len(index.index_map) == 1
    assert index.index_map[0] == 1
    assert np.allclose(index.embeddings[0], embedding / np.linalg.norm(embedding))

    with pytest.raises(ValueError):
        index.add(1, embedding)


def test_delete():
    index = Index(3)
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])

    index.add(1, embedding1)
    index.add(2, embedding2)

    index.delete(1)
    assert index.embeddings.shape == (1, 3)
    assert np.allclose(index.embeddings[0], embedding2)
    assert 1 not in index.index_map
    assert index.index_map[0] == 2

    with pytest.raises(ValueError):
        index.delete(1)


def test_save_load(tmp_path):
    index = Index(3)
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])

    index.add(1, embedding1)
    index.add(2, embedding2)

    filepath = tmp_path / "index_data.npz"
    index.save(filepath)

    loaded_index = Index(3)
    loaded_index.load(filepath)

    assert np.allclose(loaded_index.embeddings, index.embeddings)
    assert loaded_index.index_map == index.index_map


def test_query():
    index = Index(3)
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.2, 0.3, 0.4])
    embedding3 = np.array([0.3, 0.4, 0.5])

    index.add(2, embedding1)
    index.add(3, embedding2)
    index.add(4, embedding3)

    query_embedding = np.array([0.15, 0.25, 0.35])

    result = index.query(query_embedding, k=2)

    assert len(result) == 2
    assert result[0][0] == 4
    assert result[1][0] == 3
