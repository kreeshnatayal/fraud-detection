from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.fraud_detection.utils.logging import get_logger

logger = get_logger()


def build_transaction_graph(df: pd.DataFrame, edge_attrs: List[str] = None):
    try:
        import networkx as nx
    except ImportError:
        logger.error("networkx not installed — skipping graph construction.")
        return None

    edge_attrs = edge_attrs or ["card1", "addr1", "P_emaildomain"]
    G = nx.Graph()
    present = [c for c in edge_attrs if c in df.columns]

    for _, row in df[present + (["TransactionAmt"] if "TransactionAmt" in df.columns else [])].iterrows():
        nodes = [str(row[c]) for c in present if pd.notna(row[c])]
        weight = float(row["TransactionAmt"]) if "TransactionAmt" in df.columns else 1.0
        for i in range(len(nodes) - 1):
            if G.has_edge(nodes[i], nodes[i + 1]):
                G[nodes[i]][nodes[i + 1]]["weight"] += weight
            else:
                G.add_edge(nodes[i], nodes[i + 1], weight=weight)

    logger.info("Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def extract_graph_features(df: pd.DataFrame, G, edge_attrs: List[str] = None) -> pd.DataFrame:
    if G is None:
        return df
    try:
        import networkx as nx
    except ImportError:
        return df

    edge_attrs = edge_attrs or ["card1", "addr1", "P_emaildomain"]
    present = [c for c in edge_attrs if c in df.columns]

    degree = dict(G.degree())
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G, max_iter=50)

    df = df.copy()
    for col in present:
        keys = df[col].astype(str)
        df[f"{col}_graph_degree"] = keys.map(degree).fillna(0).astype(np.float32)
        df[f"{col}_graph_cluster"] = keys.map(clustering).fillna(0).astype(np.float32)
        df[f"{col}_graph_pagerank"] = keys.map(pagerank).fillna(0).astype(np.float32)

    logger.info("Graph features extracted.")
    return df


def generate_node2vec_embeddings(
    G,
    embed_dim: int = 32,
    walk_length: int = 80,
    epochs: int = 10,
) -> Optional[Dict[str, np.ndarray]]:
    if G is None:
        return None
    try:
        from node2vec import Node2Vec
        n2v = Node2Vec(G, dimensions=embed_dim, walk_length=walk_length, num_walks=10, workers=1, quiet=True)
        model = n2v.fit(window=5, min_count=1, sg=1, epochs=epochs)
        return {node: model.wv[node] for node in G.nodes() if node in model.wv}
    except Exception as exc:
        logger.warning("Node2Vec failed (%s); using random embeddings.", exc)
        return {node: np.random.randn(embed_dim).astype(np.float32) for node in G.nodes()}


def attach_embeddings(df: pd.DataFrame, embeddings: Optional[Dict], col: str, embed_dim: int = 32) -> pd.DataFrame:
    if embeddings is None or col not in df.columns:
        return df
    df = df.copy()
    zero = np.zeros(embed_dim, dtype=np.float32)
    emb_matrix = np.vstack(df[col].astype(str).map(lambda k: embeddings.get(k, zero)).tolist())
    for i in range(embed_dim):
        df[f"{col}_emb_{i}"] = emb_matrix[:, i]
    logger.info("Embeddings attached for column '%s'.", col)
    return df
