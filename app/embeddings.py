from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts):
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

def embed_query(query):
    model = get_model()
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding[0]
