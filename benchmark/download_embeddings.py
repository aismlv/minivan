import numpy as np
from tqdm import tqdm
import pandas as pd
from huggingface_hub import snapshot_download
import os
from pathlib import Path


def download_embeddings(num_partitions, local_path):
    # Download partitions of Cohere/wikipedia-22-12-en-embeddings dataset from HuggingFace Hub
    chunk_names = [f"data/train-{str(i).zfill(5)}-*.parquet" for i in range(num_partitions)]
    snapshot_download(
        repo_id="Cohere/wikipedia-22-12-en-embeddings",
        repo_type="dataset",
        allow_patterns=chunk_names,
        local_dir=local_path,
    )


def process_parquet_files(local_path):
    filepaths = os.listdir(local_path / "data")
    total_embeddings = 0

    os.makedirs(local_path / "processed", exist_ok=True)
    for i, fp in enumerate(tqdm(filepaths)):
        df = pd.read_parquet(local_path / "data" / fp)
        total_embeddings += len(df)
        embeddings = np.vstack(df["emb"].values).astype(np.float32)
        np.savez_compressed(local_path / f"processed/emb-part-{i}", embeddings=embeddings)
        print(f"Processed {i+1} files out of {len(filepaths)}: {total_embeddings} embeddings")


if __name__ == "__main__":
    num_partitions = 15
    data_path = Path("data")

    download_embeddings(num_partitions, data_path)
    process_parquet_files(data_path)
