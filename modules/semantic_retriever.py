# modules/semantic_retriever.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import sqlite3

DB_PATH = "symbolic_rag.db"

# Use a better model for improved semantic understanding
model = SentenceTransformer("all-MiniLM-L6-v2")
index = None
doc_map = {}

def build_semantic_index():
    """Build FAISS index from all documents in database."""
    global index, doc_map

    docs = []
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT rowid, title, content FROM documents;")
        docs = cursor.fetchall()
    except sqlite3.OperationalError:
        pass
    finally:
        if 'conn' in locals():
            conn.close()

    if not docs:
        print("⚠️  No documents found for semantic indexing")
        return

    # Combine title + content for richer embeddings
    texts = [f"{d[1]} {d[2]}" for d in docs]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Use Inner Product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    doc_map = {i: {"title": docs[i][1], "content": docs[i][2]} for i in range(len(docs))}
    print(f"✅ FAISS index built with {len(docs)} documents")

def semantic_search(query, top_k=3):
    """Return top-k semantically similar documents."""
    if index is None or not doc_map:
        print("⚠️  Semantic index not built. Call build_semantic_index() first.")
        return []
    
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype("float32"), top_k)
    
    results = []
    for j, i in enumerate(I[0]):
        if 0 <= i < len(doc_map):  # Valid index (prevent python negative index wrapping for FAISS sentinel -1)
            results.append((
                doc_map[i]["title"], 
                doc_map[i]["content"], 
                float(D[0][j])
            ))
    
    return results