"""Tests for evaluation, profiles and metrics (acceptance criteria A7, A11)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, silhouette_score

from segmentation import config
from segmentation.evaluate import (
    PROFILE_COLUMNS, build_profiles, compute_ari, compute_silhouette,
    load_ground_truth, name_segments, save_metrics, save_profiles,
)

METRIC_KEYS = {
    "seed", "n_customers", "chosen_k", "silhouette_by_k", "silhouette",
    "ari", "segment_sizes", "segment_names", "runtime_seconds",
}


@pytest.fixture(scope="module")
def profiles(feature_frame, labels):
    """Profiles built from the real features and the chosen labels."""
    return build_profiles(feature_frame, labels)


@pytest.fixture(scope="module")
def metrics_file(pipeline_run):
    """metrics.json as written by the pipeline run."""
    path = pipeline_run["output_dir"] / "metrics.json"
    return json.loads(path.read_text())


# ---------- A7: ARI on the real pipeline output ----------

def test_pipeline_ari_at_least_0_70(pipeline_run):
    """ARI from the files the pipeline wrote is >= 0.70.

    We merge on customer_id ourselves, so any row misalignment anywhere in the
    pipeline (e.g. shuffled feature rows) shows up as a low ARI.
    """
    segments = pd.read_csv(pipeline_run["output_dir"] / "customer_segments.csv")
    truth = pd.read_csv(pipeline_run["data_dir"] / "ground_truth.csv")
    merged = segments.merge(truth, on="customer_id", validate="one_to_one")
    assert len(merged) == config.N_CUSTOMERS
    ari = adjusted_rand_score(merged["true_segment"], merged["segment"])
    assert ari >= 0.70
    assert pipeline_run["metrics"]["ari"] == pytest.approx(ari, abs=1e-9)


def test_compute_ari_on_pipeline_output(pipeline_run):
    """compute_ari on the pipeline files agrees with the A7 threshold."""
    segments = pd.read_csv(pipeline_run["output_dir"] / "customer_segments.csv")
    truth = load_ground_truth(pipeline_run["data_dir"])
    assert list(truth.columns) == ["customer_id", "true_segment"]
    assert compute_ari(segments, truth) >= 0.70


# ---------- compute_ari / compute_silhouette sanity ----------

def small_truth():
    """A tiny ground-truth frame with three segments."""
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, 10)],
        "true_segment": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    })


def test_compute_ari_identical_is_one():
    """Labels that exactly match the truth give ARI 1."""
    truth = small_truth()
    segments = pd.DataFrame({"customer_id": truth["customer_id"],
                             "segment": [0, 0, 0, 1, 1, 1, 2, 2, 2]})
    assert compute_ari(segments, truth) == pytest.approx(1.0)


def test_compute_ari_ignores_label_ids_and_row_order():
    """Permuted label ids and shuffled rows still give ARI 1 (merge is by customer_id)."""
    truth = small_truth()
    segments = pd.DataFrame({"customer_id": truth["customer_id"],
                             "segment": [2, 2, 2, 0, 0, 0, 1, 1, 1]})
    shuffled = segments.iloc[[8, 3, 0, 5, 1, 7, 2, 6, 4]]
    assert compute_ari(shuffled, truth) == pytest.approx(1.0)


def test_compute_ari_detects_disagreement():
    """A labeling that mixes the true groups scores well below 1."""
    truth = small_truth()
    segments = pd.DataFrame({"customer_id": truth["customer_id"],
                             "segment": [0, 1, 2, 0, 1, 2, 0, 1, 2]})
    assert compute_ari(segments, truth) < 0.5


def test_compute_silhouette_matches_sklearn(X, labels):
    """compute_silhouette returns the standard silhouette score (>= 0.15, A8)."""
    value = compute_silhouette(X, labels)
    assert value == pytest.approx(silhouette_score(X, labels), abs=1e-6)
    assert value >= 0.15


# ---------- profiles and names ----------

def test_profile_columns_required():
    """PROFILE_COLUMNS includes the features the spec requires."""
    required = {"online_share", "total_spend", "order_count", "discount_rate", "grocery_share"}
    assert required <= set(PROFILE_COLUMNS)


def test_profiles_structure(profiles, labels, feature_frame):
    """One row per segment with means, size, share and name; sizes sum to n."""
    k = len(set(np.asarray(labels).tolist()))
    assert sorted(profiles.index.tolist()) == list(range(k))
    for column in PROFILE_COLUMNS + ["size", "share", "name"]:
        assert column in profiles.columns
    assert int(profiles["size"].sum()) == len(feature_frame)
    assert profiles["share"].sum() == pytest.approx(1.0)
    expected_sizes = pd.Series(labels).value_counts()
    for segment in profiles.index:
        assert int(profiles.loc[segment, "size"]) == int(expected_sizes[segment])


def test_profiles_are_raw_feature_means(profiles, labels, feature_frame):
    """Profile values are the per-segment means of the raw features."""
    expected = feature_frame.groupby(np.asarray(labels))[PROFILE_COLUMNS].mean()
    for segment in profiles.index:
        for column in PROFILE_COLUMNS:
            assert profiles.loc[segment, column] == pytest.approx(expected.loc[segment, column])


def test_profiles_have_non_empty_names(profiles):
    """Every segment gets a non-empty string name."""
    for name in profiles["name"]:
        assert isinstance(name, str) and name.strip() != ""


def test_name_segments_names_every_segment(profiles):
    """name_segments returns a non-empty name for every segment id."""
    names = name_segments(profiles)
    assert set(names) == set(profiles.index)
    assert all(isinstance(n, str) and n.strip() for n in names.values())


def test_channel_contrast_between_segments(profiles):
    """A11: some segment is mostly online (> 0.65) and some mostly store (< 0.35)."""
    assert profiles["online_share"].max() > 0.65
    assert profiles["online_share"].min() < 0.35


# ---------- metrics.json ----------

def test_metrics_has_exact_keys(metrics_file, pipeline_run):
    """metrics.json and the returned dict have exactly the keys from SPEC.md."""
    assert set(metrics_file) == METRIC_KEYS
    assert set(pipeline_run["metrics"]) == METRIC_KEYS


def test_metrics_values_are_sensible(metrics_file):
    """Each metric has a sensible value that meets the acceptance thresholds."""
    k = metrics_file["chosen_k"]
    assert metrics_file["seed"] == config.SEED
    assert metrics_file["n_customers"] == config.N_CUSTOMERS
    assert k in {4, 5, 6}
    assert set(metrics_file["silhouette_by_k"]) == {str(k) for k in range(2, 9)}
    assert metrics_file["silhouette"] >= 0.15
    assert metrics_file["ari"] >= 0.70
    assert 0 < metrics_file["runtime_seconds"] < 120


def test_metrics_segment_sizes_and_names(metrics_file):
    """segment_sizes and segment_names cover the same k segments; sizes sum to n."""
    k = metrics_file["chosen_k"]
    sizes = metrics_file["segment_sizes"]
    names = metrics_file["segment_names"]
    assert len(sizes) == k
    assert set(sizes) == set(names)
    assert sum(sizes.values()) == config.N_CUSTOMERS
    assert min(sizes.values()) >= 0.03 * config.N_CUSTOMERS  # A9
    assert all(isinstance(n, str) and n.strip() for n in names.values())


def test_pipeline_profiles_csv(pipeline_run, metrics_file):
    """segment_profiles.csv has one row per segment and the channel contrast (A11)."""
    table = pd.read_csv(pipeline_run["output_dir"] / "segment_profiles.csv")
    assert len(table) == metrics_file["chosen_k"]
    assert table["online_share"].max() > 0.65
    assert table["online_share"].min() < 0.35


# ---------- save helpers ----------

def test_save_metrics_round_trip(tmp_path):
    """save_metrics writes JSON that loads back to the same dict."""
    metrics = {"seed": 1, "silhouette_by_k": {"2": 0.3}, "segment_names": {"0": "x"}}
    save_metrics(metrics, output_dir=tmp_path)
    path = tmp_path / "metrics.json"
    assert json.loads(path.read_text()) == metrics


def test_save_profiles_writes_csv(tmp_path, profiles):
    """save_profiles writes segment_profiles.csv with one row per segment."""
    save_profiles(profiles, output_dir=tmp_path)
    loaded = pd.read_csv(tmp_path / "segment_profiles.csv")
    assert len(loaded) == len(profiles)
    assert "online_share" in loaded.columns
