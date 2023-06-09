import json
import time

import hnswlib
import numpy as np
from tqdm import tqdm

import minivan

CHUNK_SIZE = 139_004


def get_chunk(chunk_id):
    return np.load(f"data/processed/emb-part-{chunk_id}.npz")["embeddings"]


def load_data(num_items):
    num_chunks = num_items // CHUNK_SIZE
    if num_items % CHUNK_SIZE != 0:
        num_chunks += 1

    indices = np.arange(num_items).tolist()
    embeddings = np.vstack([get_chunk(i) for i in tqdm(range(num_chunks))])[:num_items, :]
    return indices, embeddings


def build_minivan_index(indices, embeddings, dim):
    start = time.time()
    index = minivan.Index(metric="dot_product", dim=dim)
    index.add_items(indices=indices, embeddings=embeddings)
    end = time.time()
    return index, end - start


def build_hnswlib_index(indices, embeddings, dim):
    # Parameters from https://github.com/nmslib/hnswlib#python-bindings-examples
    start = time.time()
    index = hnswlib.Index(space="ip", dim=dim)
    index.init_index(max_elements=len(indices), ef_construction=200, M=16)
    index.add_items(embeddings, indices)
    index.set_ef(50)
    end = time.time()
    return index, end - start


def benchmark_minivan(index, query_set, k):
    times = []

    for query_embedding in tqdm(query_set):
        start = time.time()
        _ = index.query(query_embedding, k=k)
        end = time.time()

        times.append(end - start)

    return times


def benchmark_hnswlib(index, query_set, k=10):
    times = []

    for query_embedding in tqdm(query_set):
        start = time.time()
        _ = index.knn_query(query_embedding, k=k)
        end = time.time()

        times.append(end - start)

    return times


def print_results(conf):
    print(f"Results for {conf['index']}, {conf['num_embeddings']:,}:")
    print(f"Build time: {conf['build_time']:.2f} ms")
    print(f"Mean query time: {conf['mean']:.2f} ms")
    print(f"Std query time: {conf['std']:.2f} ms")
    print(f"Max query time: {conf['max']:.2f} ms")
    print(f"Min query time: {conf['min']:.2f} ms")
    print(f"Median query time: {conf['median']:.2f} ms")
    print()


if __name__ == "__main__":
    k = 10
    dim = 768

    test_chunk_id = 14
    test_size = 5000
    test_queries = get_chunk(test_chunk_id)[:test_size]

    conf = {
        "index": "minivan",
        "metric": "dot_product",
        "dim": dim,
        "top_k": k,
    }

    for num_embeddings in [10_000, 100_000, 500_000, 1_000_000]:
        conf["num_embeddings"] = num_embeddings
        indices, embeddings = load_data(num_embeddings)

        for index_type in ["minivan", "hnswlib"]:
            conf["index"] = index_type

            if index_type == "minivan":
                index, build_time = build_minivan_index(indices, embeddings, dim)
            elif index_type == "hnswlib":
                index, build_time = build_hnswlib_index(indices, embeddings, dim)
            conf["build_time"] = build_time * 1000

            if index_type == "minivan":
                times = benchmark_minivan(index, test_queries, k=k)
            elif index_type == "hnswlib":
                times = benchmark_hnswlib(index, test_queries, k=k)
            times = [t * 1000 for t in times]
            conf["times"] = times
            conf["mean"] = np.mean(times)
            conf["std"] = np.std(times)
            conf["max"] = np.max(times)
            conf["min"] = np.min(times)
            conf["median"] = np.median(times)

            print_results(conf)

            with open("experiments/benchmark/results/results.jsonl", "a") as f:
                f.write(json.dumps(conf) + "\n")
