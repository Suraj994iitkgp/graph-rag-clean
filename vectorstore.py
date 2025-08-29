import os
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import config

# -------------------------------
# Embedding model (open source)
# -------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)
EMBEDDING_DIM = embedder.get_sentence_embedding_dimension()

# -------------------------------
# Pinecone setup with auto-fix
# -------------------------------
def init_pinecone():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    # Get list of existing indexes
    indexes = pc.list_indexes()

    if config.INDEX_NAME in [i["name"] for i in indexes]:
        info = pc.describe_index(config.INDEX_NAME)
        current_dim = info.dimension

        if current_dim != EMBEDDING_DIM:
            print(f"⚠️ Index {config.INDEX_NAME} has dim {current_dim}, "
                  f"but model requires {EMBEDDING_DIM}. Recreating index...")
            pc.delete_index(config.INDEX_NAME)
            pc.create_index(
                name=config.INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=config.CLOUD, region=config.REGION),
            )
        else:
            print(f"✅ Using existing index {config.INDEX_NAME} with correct dim {EMBEDDING_DIM}")
    else:
        print(f"ℹ️ Creating new index {config.INDEX_NAME} with dim {EMBEDDING_DIM}")
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.CLOUD, region=config.REGION),
        )

    return pc.Index(config.INDEX_NAME)

# -------------------------------
# Embedding function
# -------------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using SentenceTransformers"""
    return embedder.encode(texts, convert_to_numpy=True).tolist()

# -------------------------------
# Text chunking helper
# -------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for embedding"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -------------------------------
# Upsert documents into Pinecone
# -------------------------------
def upsert_documents(
    docs: List[Dict], chunk_size: int = 800, overlap: int = 200, namespace: str = None
) -> int:
    index = init_pinecone()
    texts_for_embeddings = []
    meta_for_embeddings = []

    for d in docs:
        doc_id = d.get("id")
        text = d.get("text", "")
        metadata = d.get("metadata", {})
        chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            texts_for_embeddings.append(chunk)
            meta_for_embeddings.append({
                "source_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                **(metadata or {})
            })

    # Generate embeddings
    vectors = embed_texts(texts_for_embeddings)

    # Batch upsert
    BATCH = 50
    for i in range(0, len(vectors), BATCH):
        batch_vectors = vectors[i : i + BATCH]
        batch_meta = meta_for_embeddings[i : i + BATCH]
        upsert_batch = []
        for j, vec in enumerate(batch_vectors):
            global_idx = i + j
            item_id = f"doc-{global_idx}"
            upsert_batch.append((item_id, vec, batch_meta[j]))
        index.upsert(vectors=upsert_batch, namespace=namespace)

    return len(vectors)

# -------------------------------
# Query Pinecone
# -------------------------------
def query_pinecone(query_text: str, top_k: int = 5, namespace: str = None):
    index = init_pinecone()
    q_emb = embed_texts([query_text])[0]
    resp = index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=namespace)
    return resp.get("matches", [])
