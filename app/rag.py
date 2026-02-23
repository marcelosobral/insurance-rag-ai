import faiss
import numpy as np
from app.embeddings import embed_query
from app.cache import check_cache, add_to_cache

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/metadata.npy"

def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError(
            "Vector index not found. Run `python -m app.ingest` to build it."
        )
    index = faiss.read_index(INDEX_PATH)
    metadata = np.load(META_PATH, allow_pickle=True)
    return index, metadata

def retrieve(query, top_k=3):
    index, metadata = load_index()

    query_embedding = embed_query(query)
    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    scores = []

    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        if distance > 1e10:
            continue

        results.append({
            "source": metadata[idx]["source"],
            "text": metadata[idx]["text"],
            "score": float(distance)
        })
        scores.append(float(distance))

    avg_score = float(np.mean(scores)) if scores else 0.0

    return {
        "results": results,
        "avg_score": avg_score
    }


def generate_answer(query, top_k=3):
    start_time = time.time()

    cached_response, cache_hit = check_cache(query)
    if cache_hit:
        return {
            **cached_response,
            "cache_hit": True,
            "latency_seconds": round(time.time() - start_time, 3)
        }

    retrieval = retrieve(query, top_k=top_k)
    contexts = [r["text"] for r in retrieval["results"]]
    sources = list(set([r["source"] for r in retrieval["results"]]))

    context_block = "\n\n".join(contexts)

    prompt = f"""
You are an insurance compliance assistant.
Answer strictly based on the provided context.
If the answer is not contained in the context, say "Insufficient information."

Context:
{context_block}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise and factual AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content
    latency = time.time() - start_time

    confidence = max(0.0, 1.0 / (1.0 + retrieval["avg_score"]))

    final_response = {
        "answer": answer,
        "sources": sources,
        "confidence": round(confidence, 3)
    }

    add_to_cache(query, final_response)

    return {
        **final_response,
        "cache_hit": False,
        "latency_seconds": round(latency, 3)
    }