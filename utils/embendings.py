""""Utility functions for embedding models."""
import json
import os

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from typing import Callable

from utils.llm import get_together_client



_local_models: dict[str, SentenceTransformer] = {}

async def get_embedding_together(text: str, model_name: str = "intfloat/multilingual-e5-large-instruct"):
    """Get embedding for a given text using Together API."""
    client = await get_together_client()
    response = await  client.embeddings.create(
        model=model_name,
        input=text
    )
    print(response.data[0].embedding)
    return response


def _get_local_model(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model once and cache it."""
    if model_name not in _local_models:
        _local_models[model_name] = SentenceTransformer(model_name, cache_folder=os.environ['MODEL_PATH'])
    return _local_models[model_name]


async def get_embedding_local(text: str, model_name: str = "BAAI/bge-base-en-v1.5"):
    """Get embedding for a given text using a local SentenceTransformer model."""
    model = _get_local_model(model_name)
    embedding = model.encode(text)
    return embedding.tolist()


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


async def create_vector_store(
    df: pd.DataFrame,
    text_columns: list[str] = ["title", "description"],
    embed_func: Callable = get_embedding_local,
    model_name: str = "BAAI/bge-base-en-v1.5",
    save_path: str = "data/vector_store.joblib",
) -> dict:
    """Create a vector store with embeddings and their corresponding DataFrame indices."""
    embeddings = []
    indices = []
    for idx, row in df.iterrows():
        combined = " ".join(f"{col}: {row[col]}" for col in text_columns)
        embedding = await embed_func(combined, model_name=model_name)
        embeddings.append(embedding)
        indices.append(idx)
    vector_store = {"embeddings": np.array(embeddings), "indices": indices}
    joblib.dump(vector_store, save_path)
    return vector_store


async def retrieve(
    query: str, top_k: int = 5, embedings_path: str = "data", embed_func: Callable = get_embedding_local
) -> list:
    """Compute the cosine similarity between two embeddings."""
    query_embedding = await embed_func(query)
    score = cosine_similarity(query_embedding)
