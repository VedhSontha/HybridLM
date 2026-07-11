# Reciprocal Rank Fusion (RRF) Formulation

Consolidated semantic and symbolic relevance scoring:
- Let $R$ be the set of retrievers, and $r(d)$ be the rank of document $d$.
- RRF Score: $RRF(d) = \sum_{r \in R} \frac{1}{k + r(d)}$.
- Penalizes documents ranking low across both models.
