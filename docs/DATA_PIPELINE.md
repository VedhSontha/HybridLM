# Ingestion Data Pipeline

Sequence of tasks processing raw documentation:
1. Document Parsing: reads text/markdown configurations.
2. Semantic Chunking: splits documents into overlapping fragments.
3. Embedding Generation: generates vector embeddings.
4. Database Upsert: updates FAISS vector store.
