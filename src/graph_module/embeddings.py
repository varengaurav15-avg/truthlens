from typing import List
import numpy as np

def embed_entities(names: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return m.encode(names)