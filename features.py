"""
Feature engineering for fraud detection.
All functions accept a DataFrame and return a new DataFrame with added columns.
"""

from __future__ import annotations

import gc
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from utils import get_logger, memory_guard

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionDT" not in df.columns:
        return df
    df = df.copy()
    dt = df["TransactionDT"]
    df["DT_hour"] = (dt // 3600) % 24
    df["DT_day"] = (dt // (3600 * 24)) % 7
    df["DT_week"] = (dt // (3600 * 24 * 7)) % 52
    df["DT_hour_sin"] = np.sin(2 * np.pi * df["DT_hour"] / 24)
    df["DT_hour_cos"] = np.cos(2 * np.pi * df["DT_hour"] / 24)
    df["DT_day_sin"] = np.sin(2 * np.pi * df["DT_day"] / 7)
    df["DT_day_cos"] = np.cos(2 * np.pi * df["DT_day"] / 7)
    df["is_night"] = ((df["DT_hour"] >= 22) | (df["DT_hour"] <= 6)).astype(np.int8)
    df["is_weekend"] = (df["DT_day"] >= 5).astype(np.int8)

    if "card1" in df.columns:
        df = df.sort_values(["card1", "TransactionDT"])
        prev_ts = df.groupby("card1")["TransactionDT"].shift(1)
        df["time_since_last_txn"] = (dt - prev_ts).fillna(1_000_000)
        df["time_since_last_txn_log"] = np.log1p(df["time_since_last_txn"])
        df["is_rapid_txn"] = (df["time_since_last_txn"] < 300).astype(np.int8)

    logger.info("Time features added.")
    return df


# ---------------------------------------------------------------------------
# Amount features
# ---------------------------------------------------------------------------

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionAmt" not in df.columns:
        return df
    df = df.copy()
    amt = df["TransactionAmt"]
    df["amount_log"] = np.log1p(amt)
    df["amount_sqrt"] = np.sqrt(amt.clip(lower=0))
    df["is_micropayment"] = (amt < 1.0).astype(np.int8)
    df["is_large_payment"] = (amt > 1_000).astype(np.int8)
    df["is_suspicious_amount"] = (amt > 5_000).astype(np.int8)
    df["amount_percentile"] = amt.rank(pct=True)

    if "card1" in df.columns:
        card_stats = df.groupby("card1")["TransactionAmt"].agg(["mean", "std"]).rename(
            columns={"mean": "_card_amt_mean", "std": "_card_amt_std"}
        )
        df = df.join(card_stats, on="card1")
        std = df["_card_amt_std"].fillna(1).replace(0, 1)
        df["amount_zscore"] = (amt - df["_card_amt_mean"].fillna(amt)) / std
        df["is_amount_outlier"] = (df["amount_zscore"].abs() > 3).astype(np.int8)
        df.drop(columns=["_card_amt_mean", "_card_amt_std"], inplace=True)

    logger.info("Amount features added.")
    return df


# ---------------------------------------------------------------------------
# Entity / behavioral features
# ---------------------------------------------------------------------------

def add_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    entity_cols = [c for c in ["card1", "card2", "addr1", "addr2"] if c in df.columns]

    for col in entity_cols:
        freq = df[col].map(df[col].value_counts())
        df[f"{col}_frequency"] = freq.fillna(0).astype(np.float32)

        if "TransactionAmt" in df.columns:
            stats = df.groupby(col)["TransactionAmt"].agg(["mean", "std", "count"])
            stats.columns = [f"{col}_amt_mean", f"{col}_amt_std", f"{col}_txn_count"]
            df = df.join(stats, on=col)

        if "isFraud" in df.columns:
            fraud_rate = df.groupby(col)["isFraud"].mean().rename(f"{col}_fraud_rate")
            df = df.join(fraud_rate, on=col)

    logger.info("Entity features added.")
    return df


# ---------------------------------------------------------------------------
# Frequency / rare-category encoding
# ---------------------------------------------------------------------------

def add_frequency_encoding(df: pd.DataFrame, rare_threshold: int = 3) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq).fillna(0).astype(np.float32)
        df[f"{col}_is_rare"] = (df[f"{col}_freq"] < rare_threshold).astype(np.int8)
    logger.info("Frequency encoding added.")
    return df


# ---------------------------------------------------------------------------
# Target-mean encoding (requires isFraud column)
# ---------------------------------------------------------------------------

def add_target_encoding(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    if "isFraud" not in df.columns:
        return df
    df = df.copy()
    cols = cols or [c for c in ["card1", "card2", "addr1", "P_emaildomain", "DeviceInfo"] if c in df.columns]
    global_mean = df["isFraud"].mean()
    for col in cols:
        means = df.groupby(col)["isFraud"].mean()
        df[f"{col}_target_enc"] = df[col].map(means).fillna(global_mean).astype(np.float32)
    logger.info("Target encoding added.")
    return df


# ---------------------------------------------------------------------------
# Domain-based features (email / device)
# ---------------------------------------------------------------------------

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for email_col in ["P_emaildomain", "R_emaildomain"]:
        if email_col not in df.columns:
            continue
        df[f"{email_col}_domain"] = df[email_col].str.split(".").str[-1].fillna("unknown")
        df[f"{email_col}_is_free"] = df[email_col].isin(
            ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
        ).astype(np.int8)

    if "DeviceInfo" in df.columns:
        df["device_os"] = df["DeviceInfo"].str.split(" ").str[0].fillna("unknown")
        df["is_mobile"] = df["DeviceInfo"].str.lower().str.contains("android|ios|mobile", na=False).astype(np.int8)

    logger.info("Domain features added.")
    return df


# ---------------------------------------------------------------------------
# Rolling / lag features
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    if "TransactionDT" not in df.columns or "card1" not in df.columns:
        return df
    windows = windows or [1, 3, 7, 14]
    df = df.copy().sort_values(["card1", "TransactionDT"])

    for w in windows:
        if "TransactionAmt" in df.columns:
            df[f"rolling_amt_mean_{w}d"] = (
                df.groupby("card1")["TransactionAmt"]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
                .astype(np.float32)
            )
        df[f"rolling_txn_count_{w}d"] = (
            df.groupby("card1")["TransactionDT"]
            .transform(lambda x: x.rolling(w, min_periods=1).count())
            .astype(np.float32)
        )

    logger.info("Rolling features added.")
    return df


def add_lag_features(df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
    if "TransactionDT" not in df.columns or "card1" not in df.columns:
        return df
    lags = lags or [1, 3, 7]
    df = df.copy().sort_values(["card1", "TransactionDT"])

    for lag in lags:
        if "TransactionAmt" in df.columns:
            df[f"lag_amt_{lag}"] = (
                df.groupby("card1")["TransactionAmt"].shift(lag).astype(np.float32)
            )
        if "isFraud" in df.columns:
            df[f"lag_fraud_{lag}"] = (
                df.groupby("card1")["isFraud"].shift(lag).astype(np.float32)
            )

    logger.info("Lag features added.")
    return df


# ---------------------------------------------------------------------------
# Anomaly score features
# ---------------------------------------------------------------------------

def add_anomaly_features(df: pd.DataFrame, sample_size: int = 10_000) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(
        ["isFraud", "TransactionID"]
    ).tolist()
    if len(num_cols) < 2:
        return df

    subset = df[num_cols].fillna(0)
    n = min(sample_size, len(df))

    try:
        iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
        iso.fit(subset.sample(n=n, random_state=42))
        df["isolation_score"] = iso.score_samples(subset).astype(np.float32)
    except Exception as exc:
        logger.warning("IsolationForest failed: %s", exc)

    try:
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
        lof.fit(subset.sample(n=n, random_state=42))
        df["lof_score"] = lof.score_samples(subset).astype(np.float32)
    except Exception as exc:
        logger.warning("LocalOutlierFactor failed: %s", exc)

    logger.info("Anomaly features added.")
    return df


# ---------------------------------------------------------------------------
# Graph features (NetworkX-based)
# ---------------------------------------------------------------------------

def build_transaction_graph(df: pd.DataFrame, edge_attrs: List[str] = None) -> "nx.Graph":
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


# ---------------------------------------------------------------------------
# Node2Vec embeddings
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Master feature engineering pipeline
# ---------------------------------------------------------------------------

def run_feature_engineering(df: pd.DataFrame, config) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    steps = [
        ("time", lambda d: add_time_features(d)),
        ("amount", lambda d: add_amount_features(d)),
        ("entity", lambda d: add_entity_features(d)),
        ("frequency", lambda d: add_frequency_encoding(d, rare_threshold=config.RARE_THRESHOLD)),
        ("target_enc", lambda d: add_target_encoding(d)),
        ("domain", lambda d: add_domain_features(d)),
        ("rolling", lambda d: add_rolling_features(d, windows=config.ROLLING_WINDOWS)),
        ("lag", lambda d: add_lag_features(d, lags=config.LAG_PERIODS)),
        ("anomaly", lambda d: add_anomaly_features(d, sample_size=config.SAMPLE_SIZE)),
    ]

    for name, fn in steps:
        logger.info("Feature step: %s", name)
        with memory_guard(threshold_pct=config.CRITICAL_MEMORY_THRESHOLD * 100):
            df = fn(df)
        gc.collect()

    return df
