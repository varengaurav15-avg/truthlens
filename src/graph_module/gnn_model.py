# Placeholder: use graph_query for lookup; GNN optional for research
def get_contradiction_score(claim_embedding, subgraph_embedding: list) -> float:
    import numpy as np
    if not subgraph_embedding:
        return 0.5
    emb = np.array(subgraph_embedding).mean(axis=0)
    sim = np.dot(claim_embedding, emb) / (np.linalg.norm(claim_embedding) * np.linalg.norm(emb) + 1e-9)
    return float(np.clip(0.5 + 0.5 * sim, 0, 1))