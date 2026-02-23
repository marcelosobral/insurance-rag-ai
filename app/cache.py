import json
import os
import numpy as np
from app.embeddings import embed_query

CACHE_PATH = "vector_store/cache.json"

def load_cache():
    if not os.path.exists(CACHE_PATH):
        return []
    with open(CACHE_PATH, "r") as f:
        return json.load(f)

def save_cache(cache):
    cache_dir = os.path.dirname(CACHE_PATH)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def check_cache(query, threshold=0.92):
    cache = load_cache()
    query_embedding = embed_query(query)

    for item in cache:
        cached_embedding = np.array(item["embedding"])
        sim = cosine_similarity(query_embedding, cached_embedding)
        if sim > threshold:
            return item["response"], True

    return None, False

def add_to_cache(query, response):
    cache = load_cache()
    embedding = embed_query(query).tolist()

    cache.append({
        "query": query,
        "embedding": embedding,
        "response": response
    })

    save_cache(cache)