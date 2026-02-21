import os
import faiss
import numpy as np
from app.embeddings import embed_texts

DATA_DIR = "data"
INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/metadata.npy"

def load_documents():
    docs = []
    metadata = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_DIR, filename), "r") as f:
                text = f.read()
                chunks = chunk_text(text)
                for chunk in chunks:
                    docs.append(chunk)
                    metadata.append({
                        "source": filename,
                        "text": chunk
                    })

    return docs, metadata


def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def build_index():
    docs, metadata = load_documents()
    embeddings = embed_texts(docs)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, metadata)

    print("Index built successfully!")

if __name__ == "__main__":
    build_index()
