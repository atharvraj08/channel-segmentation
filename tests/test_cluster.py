"""Tests for choosing k and fitting segments (acceptance criteria A6, A8, A9, A10)."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from segmentation import config
from segmentation.cluster import fit_segments, save_segments


# ---------- A6: choose_k ----------

def test_choose_k_scores_every_k(k_result):
    """choose_k returns a silhouette score for every k in 2..8."""
    _, scores = k_result
    assert set(scores) == set(range(2, 9))
    assert set(scores) == set(config.K_RANGE)
    for score in scores.values():
        assert -1.0 <= float(score) <= 1.0


def test_chosen_k_is_five_plus_minus_one(k_result):
    """The chosen k is 4, 5 or 6 (catches a hard-coded k = 2)."""
    best_k, scores = k_result
    assert best_k in {4, 5, 6}
    assert best_k in scores


# ---------- labels ----------

def test_labels_are_ints_zero_to_k_minus_one(X, k_result, labels):
    """Labels are integers 0..k-1, one per row, and every cluster is used."""
    best_k, _ = k_result
    labels = np.asarray(labels)
    assert labels.shape == (len(X),)
    assert np.issubdtype(labels.dtype, np.integer)
    assert set(labels.tolist()) == set(range(best_k))


def test_fit_segments_is_deterministic(X, k_result, labels):
    """The same seed gives exactly the same labels."""
    best_k, _ = k_result
    again = fit_segments(X, best_k, seed=config.SEED)
    assert np.array_equal(np.asarray(again), np.asarray(labels))


def test_labels_follow_the_rows_of_X(X, k_result, labels):
    """Fitting on reversed rows gives the reversed partition (no internal shuffling)."""
    best_k, _ = k_result
    reversed_labels = fit_segments(X[::-1], best_k, seed=config.SEED)
    assert adjusted_rand_score(np.asarray(reversed_labels)[::-1], labels) >= 0.80


def test_labels_recover_true_segments(feature_frame, ground_truth, labels):
    """Labels aligned by customer id agree with the hidden truth (ARI >= 0.70)."""
    truth = ground_truth.set_index("customer_id").loc[feature_frame.index, "true_segment"]
    assert adjusted_rand_score(truth.to_numpy(), labels) >= 0.70


# ---------- A8, A9, A10 ----------

def test_silhouette_at_least_0_15(X, labels):
    """The chosen model has silhouette >= 0.15."""
    assert silhouette_score(X, labels) >= 0.15


def test_every_cluster_at_least_3_percent(labels):
    """No predicted segment is smaller than 3% of customers."""
    shares = pd.Series(labels).value_counts(normalize=True)
    assert shares.min() >= 0.03


def test_stable_across_seeds(X, k_result, labels):
    """Re-clustering with seed 7 agrees with seed 42 (ARI >= 0.80)."""
    best_k, _ = k_result
    other = fit_segments(X, best_k, seed=7)
    assert adjusted_rand_score(labels, other) >= 0.80


# ---------- save_segments ----------

def test_save_segments_writes_csv(tmp_path):
    """save_segments writes customer_id, segment in the given order and returns that frame."""
    ids = ["C00001", "C00002", "C00003", "C00004"]
    labels = np.array([1, 0, 1, 2])
    returned = save_segments(ids, labels, output_dir=tmp_path)
    path = tmp_path / "customer_segments.csv"
    assert path.exists()
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == ["customer_id", "segment"]
    assert list(loaded["customer_id"]) == ids
    assert list(loaded["segment"]) == [1, 0, 1, 2]
    assert list(returned.columns) == ["customer_id", "segment"]
    assert list(returned["segment"]) == [1, 0, 1, 2]
