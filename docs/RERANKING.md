# LLM Reciprocal Rank Fusion (RRF) Guide

Fusing sparse keyword match with dense semantic vector search:
- Formula: $RRF(d) = \sum_{m \in M} rac{1}{k + r_m(d)}$
- Defaults: $k = 60$
- Combines semantic outputs with FTS5 search.
