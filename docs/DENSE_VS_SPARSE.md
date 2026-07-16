# Dense Vector vs. Sparse Keyword Search

Comparing search components in Neuro-Symbolic RAG:
- Dense: FAISS L2 Euclidean distance matcher; captures semantic context.
- Sparse: SQLite FTS5 BM25 matcher; captures exact keyword matches.
