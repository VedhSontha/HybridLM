# modules/symbolic_retriever.py
import sqlite3
from modules.preprocess import TextPreprocessor

tp = TextPreprocessor()
DB_PATH = "symbolic_rag.db"

def init_symbolic_db():
    """Initialize SQLite FTS5 database with entity-aware schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS documents 
    USING fts5(
        title, 
        content,
        tokenize='porter unicode61'
    );
    """)
    conn.commit()
    conn.close()
    print("✅ Symbolic database initialized")

def insert_documents(docs):
    """Insert list of (title, content) tuples with entity restoration."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Apply entity restoration to preserve proper entity names
    processed_docs = [
        (tp.restore_entities(title), tp.restore_entities(content))
        for title, content in docs
    ]
    
    cursor.executemany("INSERT INTO documents (title, content) VALUES (?, ?);", processed_docs)
    conn.commit()
    conn.close()
    print(f"✅ Inserted {len(docs)} documents")

def symbolic_search(query, top_k=3):
    """Entity-aware symbolic keyword search using FTS5."""
    # Restore entities in query (e.g., "medicalgpt" → "MedicalGPT")
    q_restored = tp.restore_entities(query)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Split into terms, keeping important entities
    terms = [t.strip() for t in q_restored.split() if len(t.strip()) > 1]
    if not terms:
        return []
    
    # Build FTS5 match expression (OR all terms for broader recall)
    match_expr = " OR ".join([f'"{term}"' for term in terms])
    
    try:
        sql = """
            SELECT title, content, rank
            FROM documents
            WHERE documents MATCH ?
            ORDER BY rank
            LIMIT ?;
        """
        results = cursor.execute(sql, (match_expr, top_k)).fetchall()
    except sqlite3.OperationalError as e:
        print(f"⚠️  FTS5 search error: {e}")
        results = []
    
    conn.close()
    return [(r[0], r[1]) for r in results]