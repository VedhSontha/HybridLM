# Reciprocal Rank Fusion Retrieval

Formulating fused relevance across sparse and dense scores:
- Document rank score: $R(d) = \frac{1}{60 + r_{vector}(d)} + \frac{1}{60 + r_{keyword}(d)}$.
- Normalizes disparate scoring outputs for target extraction.
