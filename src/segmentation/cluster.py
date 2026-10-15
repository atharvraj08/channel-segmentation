"""Choose the number of segments and assign every customer to one.

Works only on the scaled feature matrix X; it never sees the hidden answer key,
so k is picked purely from the data's own structure (silhouette score).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from segmentation import config


def _kmeans(k: int, seed: int) -> KMeans:
    """A KMeans model with the project's standard settings.

    n_init=10 restarts from 10 random starting points and keeps the best,
    which makes the result stable across seeds.
    """
    return KMeans(n_clusters=k, n_init=10, random_state=seed)


def choose_k(X: np.ndarray, k_range=config.K_RANGE,
             seed: int = config.SEED) -> tuple[int, dict[int, float]]:
    """Fit KMeans for every k in k_range and return (best_k, {k: silhouette}).

    Silhouette compares how close each customer is to its own segment versus
    the nearest other one; the k with the highest average wins.
    """
    scores: dict[int, float] = {}
    for k in k_range:
        labels = _kmeans(k, seed).fit_predict(X)
        scores[int(k)] = float(silhouette_score(X, labels))
    best_k = max(scores, key=scores.get)
    return best_k, scores


def fit_segments(X: np.ndarray, k: int, seed: int = config.SEED) -> np.ndarray:
    """Return integer labels 0..k-1 (one per row of X). Deterministic for a given seed."""
    labels = _kmeans(k, seed).fit_predict(X)
    return labels.astype(int)


def save_segments(customer_ids, labels: np.ndarray,
                  output_dir: Path = config.OUTPUT_DIR) -> pd.DataFrame:
    """Write output_dir/customer_segments.csv (customer_id, segment) and return that frame."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = pd.DataFrame({
        "customer_id": list(customer_ids),
        "segment": np.asarray(labels).astype(int),
    })
    segments.to_csv(output_dir / "customer_segments.csv", index=False)
    return segments
