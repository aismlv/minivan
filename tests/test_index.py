import numpy as np
import pytest

from minivan import Index


@pytest.fixture
def embedding1():
    return np.array([0.1, 0.2, 0.3])


@pytest.fixture
def embedding2():
    return np.array([0.2, 0.3, 0.4])


@pytest.fixture
def embedding3():
    return np.array([0.3, 0.4, 0.5])


@pytest.mark.parametrize(
    "indices, embeddings",
    [
        ([1, 2], [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]),
        ([1, 2], np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])),
        (np.array([1, 2]), [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]),
        (np.array([1, 2]), np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])),
    ],
)
def test_add_items(indices, embeddings):
    index = Index(3, metric="dot_product")
    index.add_items(indices, embeddings)

    assert index.embeddings.shape == (2, 3)
    assert len(index.index_map) == 2
    assert index.index_map[0] == 1
    assert np.allclose(index.embeddings[0], np.array([0.1, 0.2, 0.3]))


def test_invalid_add_items(embedding1, embedding2):
    index = Index(3, metric="dot_product")
    embeddings = np.vstack([embedding1, embedding2])

    with pytest.raises(KeyError):
        index.add_items([1], [embedding1])
        index.add_items([1], [embedding1])
    with pytest.raises(TypeError):
        index.add_items("not indices", [embedding1])
    with pytest.raises(TypeError):
        index.add_items(["2"], [embedding1])
    with pytest.raises(ValueError):
        index.add_items([3, 4], [embedding1, np.array([0.1, 0.2])])
    with pytest.raises(ValueError):
        index.add_items([3], embeddings)


def test_add_items_cosine(embedding1):
    index = Index(3, metric="cosine")

    index.add_items([1], [embedding1])
    assert index.embeddings.shape == (1, 3)
    assert len(index.index_map) == 1
    assert index.index_map[0] == 1
    assert np.allclose(index.embeddings[0], embedding1 / np.linalg.norm(embedding1))

    with pytest.raises(KeyError):
        index.add_items([1], [embedding1])


def test_delete_items(embedding1, embedding2):
    index = Index(3)
    index.add_items([1, 2], [embedding1, embedding2])

    index.delete_items([1])
    assert index.embeddings.shape == (1, 3)
    assert np.allclose(index.embeddings[0], embedding2)
    assert 1 not in index.index_map
    assert index.index_map[0] == 2

    with pytest.raises(KeyError):
        index.delete_items([1])


def test_save_load(tmp_path, embedding1, embedding2):
    index = Index(3, metric="dot_product")
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


def test_load_from_file(tmp_path, embedding1, embedding2):
    index = Index(3, metric="dot_product")
    index.add_items([1, 2], [embedding1, embedding2])

    filepath = tmp_path / "index_data.npz"
    index.save(filepath)

    loaded_index = Index.from_file(filepath)
    assert np.allclose(loaded_index.embeddings, index.embeddings)
    assert loaded_index.index_map == index.index_map


def test_query_dot_product(embedding1, embedding2, embedding3):
    query_embedding = np.array([0.15, 0.25, 0.35])

    index = Index(3, metric="dot_product")
    index.add_items([2, 3, 4], [embedding1, embedding2, embedding3])
    result = index.query(query_embedding, k=2)

    assert len(result) == 2
    assert result[0][0] == 4
    assert result[1][0] == 3

    with pytest.raises(ValueError):
        index.query(query_embedding, k=0)
    with pytest.raises(ValueError):
        index.query(query_embedding, k=-1)
    with pytest.raises(ValueError):
        index.query(query_embedding, k=4)
    with pytest.raises(ValueError):
        index.query(np.array([0.1, 0.2, 0.3, 0.4]), k=2)


def test_query_cosine(embedding1, embedding2, embedding3):
    query_embedding = np.array([0.15, 0.25, 0.35])

    index = Index(3, metric="cosine")
    index.add_items([2, 3, 4], [embedding1, embedding2, embedding3])
    result = index.query(query_embedding, k=2)

    assert len(result) == 2
    assert result[0][0] == 3
    assert result[1][0] == 2


def test_query_item(embedding1, embedding2, embedding3):
    index = Index(3, metric="dot_product")
    index.add_items([2, 3, 4], [embedding1, embedding2, embedding3])
    result = index.query_item(2, k=2)

    assert len(result) == 2
    assert result[0][0] == 4
    assert 2 not in [item[0] for item in result]


def test_len(embedding1, embedding2):
    index = Index(3)
    assert len(index) == 0

    index.add_items([1, 2], [embedding1, embedding2])
    assert len(index) == 2

    index.delete_items([1])
    assert len(index) == 1


def test_get_item(embedding1, embedding2):
    index = Index(3)
    index.add_items([1, 2], [embedding1, embedding2])
    assert np.allclose(index.get_items([1]), embedding1)
    assert np.allclose(index.get_items([1, 2]), np.vstack([embedding1, embedding2]))

    with pytest.raises(KeyError):
        index.get_items([3])
