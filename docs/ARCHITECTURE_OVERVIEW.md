# HybridLM Architecture Overview

```
Raw Documents
     |
     v
[Chunker] --> [BERT Embedder] --> [FAISS Index]
     |                                  |
     v                                  v
[FTS5 Indexer] -----> [RRF Fusion] <----+
                           |
                           v
                  [Representation Bridge]
                     BERT 768d -> LLaMA 4096d
                           |
                           v
                   [LLaMA 3.2 (NF4)]
                           |
                           v
                   Incident Report
```
