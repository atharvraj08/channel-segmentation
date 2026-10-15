"""Tests for feature engineering and scaling (acceptance criteria A4, A5)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from segmentation import cluster, features
from segmentation.features import (
    FEATURE_COLUMNS, LOG_COLUMNS, MODEL_FEATURES,
    build_features, load_raw_data, scale_features,
)
from segmentation.generate_data import generate_data, save_data

SHARE_COLUMNS = ["online_share", "discount_rate", "top_category_share",
                 "grocery_share", "email_open_rate"]


def tiny_inputs():
    """Three hand-built customers whose features are easy to compute by hand.

    C00001 buys only online, C00002 only in store, C00003 in both channels.
    Engagement rows are deliberately out of order to check the index gets sorted.
    """
    transactions = pd.DataFrame(
        [
            ("C00001", "2026-09-01", "online", 100.0, "electronics", True),
            ("C00001", "2026-09-10", "online", 50.0, "grocery", False),
            ("C00002", "2026-09-20", "store", 30.0, "grocery", True),
            ("C00002", "2026-09-25", "store", 10.0, "grocery", False),
            ("C00002", "2026-09-30", "store", 60.0, "home", False),
            ("C00003", "2026-08-31", "online", 40.0, "apparel", False),
            ("C00003", "2026-09-30", "store", 60.0, "apparel", True),
        ],
        columns=["customer_id", "date", "channel", "amount", "category", "used_discount"],
    )
    engagement = pd.DataFrame(
        [
            ("C00003", 0.5, 1.0, 2.0, False, 0),
            ("C00001", 0.9, 10.0, 6.0, True, 20),
            ("C00002", 0.1, 0.0, 0.5, True, 0),
        ],
        columns=["customer_id", "email_open_rate", "app_sessions_per_month",
                 "web_sessions_per_month", "loyalty_member", "days_since_last_visit"],
    )
    return transactions, engagement


@pytest.fixture
def tiny_features():
    """Features built from the tiny hand-made example."""
    return build_features(*tiny_inputs())


# ---------- A4 on the real data ----------

def test_one_row_per_customer(feature_frame, engagement):
    """Index is the sorted list of customer ids from engagement."""
    assert feature_frame.index.is_unique
    assert list(feature_frame.index) == sorted(engagement["customer_id"])


def test_columns_are_feature_columns(feature_frame):
    """Columns are exactly FEATURE_COLUMNS in order, all float."""
    assert list(feature_frame.columns) == FEATURE_COLUMNS
    assert all(pd.api.types.is_float_dtype(t) for t in feature_frame.dtypes)


def test_feature_lists_are_consistent():
    """MODEL_FEATURES and LOG_COLUMNS only use names from FEATURE_COLUMNS."""
    assert len(MODEL_FEATURES) > 0
    assert set(MODEL_FEATURES) <= set(FEATURE_COLUMNS)
    assert set(LOG_COLUMNS) <= set(FEATURE_COLUMNS)


def test_no_nan_or_inf(feature_frame):
    """Every value is finite."""
    assert np.isfinite(feature_frame.to_numpy(dtype=float)).all()


def test_shares_and_rates_in_unit_interval(feature_frame):
    """Shares and rates lie in [0, 1]; loyalty_member is 0/1."""
    for column in SHARE_COLUMNS:
        assert feature_frame[column].between(0, 1).all(), column
    assert set(feature_frame["loyalty_member"].unique()) <= {0.0, 1.0}


def test_counts_are_consistent(feature_frame):
    """Channel order counts add up to the total and everyone ordered at least once."""
    total = feature_frame["online_orders"] + feature_frame["store_orders"]
    assert np.allclose(total, feature_frame["order_count"])
    assert (feature_frame["order_count"] >= 1).all()
    assert (feature_frame["total_spend"] > 0).all()
    assert (feature_frame["recency_days"] >= 0).all()


def test_raw_features_are_not_already_scaled(feature_frame):
    """build_features returns raw values; otherwise a missing scaling step would go unnoticed."""
    assert feature_frame["total_spend"].mean() > 10
    assert feature_frame["order_count"].mean() > 1


def test_scaled_shape(feature_frame, X):
    """X has one row per customer and one column per model feature."""
    assert X.shape == (len(feature_frame), len(MODEL_FEATURES))


def test_scaled_mean_zero_std_one(X):
    """Every scaled column has mean ~0 and std ~1 (catches a removed scaling step)."""
    assert np.abs(X.mean(axis=0)).max() < 1e-6
    assert np.abs(X.std(axis=0) - 1).max() < 1e-3


def test_scaling_matches_log_then_standardize(feature_frame, X):
    """X equals log1p on LOG_COLUMNS followed by standardization, column by column."""
    for i, column in enumerate(MODEL_FEATURES):
        values = feature_frame[column].to_numpy(dtype=float)
        if column in LOG_COLUMNS:
            values = np.log1p(values)
        expected = (values - values.mean()) / values.std()
        assert np.allclose(X[:, i], expected, atol=1e-6), column


def test_scaling_keeps_row_order(feature_frame, X):
    """Reversing the input rows reverses X: rows are never reordered internally."""
    reversed_X, _ = scale_features(feature_frame.iloc[::-1])
    assert np.allclose(reversed_X, X[::-1])


# ---------- hand-checked tiny example ----------

def test_tiny_index_sorted(tiny_features):
    """The index is sorted customer ids even though engagement was not."""
    assert list(tiny_features.index) == ["C00001", "C00002", "C00003"]
    assert list(tiny_features.columns) == FEATURE_COLUMNS


def test_tiny_online_only_customer(tiny_features):
    """An online-only customer has online_share 1 and zero store orders."""
    row = tiny_features.loc["C00001"]
    assert row["online_share"] == pytest.approx(1.0)
    assert row["online_orders"] == pytest.approx(2)
    assert row["store_orders"] == pytest.approx(0)


def test_tiny_store_only_customer(tiny_features):
    """A store-only customer has online_share 0 and zero online orders."""
    row = tiny_features.loc["C00002"]
    assert row["online_share"] == pytest.approx(0.0)
    assert row["online_orders"] == pytest.approx(0)
    assert row["store_orders"] == pytest.approx(3)


def test_tiny_no_nan(tiny_features):
    """Zero-order channels never produce NaN."""
    assert np.isfinite(tiny_features.to_numpy(dtype=float)).all()


def test_tiny_hand_checked_values(tiny_features):
    """Spend, counts, rates and shares match values worked out by hand."""
    expected = {
        "C00001": dict(total_spend=150, order_count=2, avg_basket=75, discount_rate=0.5,
                       recency_days=20, n_categories=2, top_category_share=100 / 150,
                       grocery_share=50 / 150),
        "C00002": dict(total_spend=100, order_count=3, avg_basket=100 / 3, discount_rate=1 / 3,
                       recency_days=0, n_categories=2, top_category_share=0.6,
                       grocery_share=0.4),
        "C00003": dict(total_spend=100, order_count=2, avg_basket=50, discount_rate=0.5,
                       recency_days=0, n_categories=1, top_category_share=1.0,
                       grocery_share=0.0, online_share=0.4),
    }
    for customer, values in expected.items():
        for column, value in values.items():
            assert tiny_features.loc[customer, column] == pytest.approx(value), (customer, column)


def test_tiny_engagement_copied(tiny_features):
    """Engagement columns are copied and loyalty becomes 1.0/0.0."""
    row = tiny_features.loc["C00001"]
    assert row["email_open_rate"] == pytest.approx(0.9)
    assert row["app_sessions"] == pytest.approx(10.0)
    assert row["web_sessions"] == pytest.approx(6.0)
    assert row["loyalty_member"] == pytest.approx(1.0)
    assert tiny_features.loc["C00003", "loyalty_member"] == pytest.approx(0.0)


# ---------- loading ----------

def test_load_raw_data_does_not_need_ground_truth(tmp_path):
    """load_raw_data works even when ground_truth.csv has been deleted."""
    tx, eng, truth = generate_data(n_customers=100, seed=11)
    save_data(tx, eng, truth, data_dir=tmp_path)
    (tmp_path / "ground_truth.csv").unlink()
    loaded_tx, loaded_eng = load_raw_data(tmp_path)
    assert len(loaded_tx) == len(tx)
    assert len(loaded_eng) == len(eng)
    assert "true_segment" not in loaded_tx.columns
    assert "true_segment" not in loaded_eng.columns


# ---------- A5: no label leakage in model code ----------

@pytest.mark.parametrize("module", [features, cluster])
def test_model_code_never_mentions_labels(module):
    """features.py and cluster.py never contain the strings ground_truth or true_segment."""
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "ground_truth" not in source
    assert "true_segment" not in source
