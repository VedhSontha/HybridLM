import numpy as np
from typing import List

def validate_embeddings(vectors: List[np.ndarray], expected_dim: int = 768) -> bool:
    """Checks all embedding vectors match the expected dimensionality."""
    for i, vec in enumerate(vectors):
        if vec.shape[0] != expected_dim:
            print(f"Vector {i} has dim {vec.shape[0]}, expected {expected_dim}")
            return False
    return True

if __name__ == '__main__':
    test_vecs = [np.random.randn(768) for _ in range(10)]
    print(f"Validation: {validate_embeddings(test_vecs)}")
