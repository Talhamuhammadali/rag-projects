""""Utility functions for embedding models."""
import json
import os

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.llm import get_together_client



async def get_embedding_together(text: str, model: str = "intfloat/multilingual-e5-large-instruct"):
    """Get embedding for a given text using Together API."""
    client = await get_together_client()
    response = await  client.embeddings.create(
        model=model,
        input=text
    )
    print(response.data[0].embedding)
    return response
    