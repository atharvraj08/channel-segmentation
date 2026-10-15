"""Tests for the synthetic data generator (acceptance criteria A1, A2, A3)."""

import re

import numpy as np
import pandas as pd

from segmentation import config
from segmentation.generate_data import generate_data, save_data

TRANSACTION_COLUMNS = ["customer_id", "date", "channel", "amount", "category", "used_discount"]
ENGAGEMENT_COLUMNS = [
    "customer_id", "email_open_rate", "app_sessions_per_month",
    "web_sessions_per_month", "loyalty_member", "days_since_last_visit",
]
GROUND_TRUTH_COLUMNS = ["customer_id", "true_segment"]


# ---------- A1: shape, columns, values ----------

def test_exact_columns(transactions, engagement, ground_truth):
    """Each table has exactly the columns from SPEC.md, in order."""
    assert list(transactions.columns) == TRANSACTION_COLUMNS
    assert list(engagement.columns) == ENGAGEMENT_COLUMNS
    assert list(ground_truth.columns) == GROUND_TRUTH_COLUMNS


def test_customer_count_and_ids(engagement, ground_truth):
    """Exactly N_CUSTOMERS customers with ids C00001, C00002, ..."""
    expected_ids = [f"C{i:05d}" for i in range(1, config.N_CUSTOMERS + 1)]
    assert config.N_CUSTOMERS == 5000
    assert sorted(engagement["customer_id"]) == expected_ids
    assert sorted(ground_truth["customer_id"]) == expected_ids


def test_one_engagement_row_per_customer(engagement):
    """Engagement has no duplicate customers."""
    assert engagement["customer_id"].is_unique
    assert len(engagement) == config.N_CUSTOMERS


def test_every_customer_has_a_transaction(transactions, engagement):
    """Every customer appears in transactions, and no unknown customer does."""
    assert set(transactions["customer_id"]) == set(engagement["customer_id"])


def test_no_missing_values(transactions, engagement, ground_truth):
    """No NaN anywhere in the three tables."""
    for frame in (transactions, engagement, ground_truth):
        assert not frame.isna().any().any()


def test_amounts_positive_and_rounded(transactions):
    """Amounts are strictly positive floats rounded to 2 decimals."""
    amounts = transactions["amount"].to_numpy(dtype=float)
    assert (amounts > 0).all()
    assert np.allclose(amounts, np.round(amounts, 2))


def test_channels_and_categories_valid(transactions):
    """Channel is online/store and both appear; category is one of CATEGORIES."""
    assert set(transactions["channel"]) == {"online", "store"}
    assert set(transactions["category"]) <= set(config.CATEGORIES)
    assert pd.api.types.is_bool_dtype(transactions["used_discount"])


def test_dates_within_history_window(transactions):
    """Dates are YYYY-MM-DD strings inside the 365 days ending at END_DATE."""
    assert transactions["date"].map(lambda d: bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d))).all()
    dates = pd.to_datetime(transactions["date"], format="%Y-%m-%d")
    end = pd.Timestamp(config.END_DATE)
    # Allow END_DATE - 365 days so either reading of "365 days ending END_DATE" passes.
    start = end - pd.Timedelta(days=config.N_DAYS)
    assert dates.max() <= end
    assert dates.min() >= start


def test_transactions_sorted(transactions):
    """Transactions are sorted by customer_id then date."""
    keys = transactions[["customer_id", "date"]].reset_index(drop=True)
    expected = keys.sort_values(["customer_id", "date"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(keys, expected)


def test_engagement_value_ranges(engagement):
    """Rates in [0, 1], sessions >= 0, loyalty is bool, recency is int 0..365."""
    assert engagement["email_open_rate"].between(0, 1).all()
    assert (engagement["app_sessions_per_month"] >= 0).all()
    assert (engagement["web_sessions_per_month"] >= 0).all()
    assert pd.api.types.is_bool_dtype(engagement["loyalty_member"])
    assert pd.api.types.is_integer_dtype(engagement["days_since_last_visit"])
    assert engagement["days_since_last_visit"].between(0, 365).all()


def test_five_true_segments_each_at_least_8_percent(ground_truth, segment_names):
    """All 5 planted segments exist, each >= 8% of customers, with unequal sizes."""
    shares = ground_truth["true_segment"].value_counts(normalize=True)
    assert set(shares.index) == set(segment_names)
    assert config.N_TRUE_SEGMENTS == 5
    assert (shares >= 0.08).all()
    assert shares.nunique() > 1  # spec asks for unequal sizes


# ---------- A2: reproducibility ----------

def test_same_seed_gives_identical_data(generated):
    """Generating again with the default seed reproduces every table exactly."""
    again = generate_data(n_customers=config.N_CUSTOMERS, seed=config.SEED)
    for first, second in zip(generated, again):
        pd.testing.assert_frame_equal(first, second)


def test_different_seed_gives_different_data():
    """A different seed must change the data (catches a generator ignoring its seed)."""
    tx_a, eng_a, _ = generate_data(n_customers=300, seed=1)
    tx_b, eng_b, _ = generate_data(n_customers=300, seed=2)
    assert not tx_a.equals(tx_b)
    assert not eng_a.equals(eng_b)


def test_n_customers_argument_respected():
    """A small n_customers gives exactly that many customers."""
    tx, eng, truth = generate_data(n_customers=300, seed=3)
    assert len(eng) == 300
    assert len(truth) == 300
    assert tx["customer_id"].nunique() == 300


# ---------- A3: no label leakage ----------

def test_no_true_segment_column_in_inputs(transactions, engagement):
    """The model's input tables do not carry the label column."""
    assert "true_segment" not in transactions.columns
    assert "true_segment" not in engagement.columns


def test_no_segment_names_in_inputs(transactions, engagement, segment_names):
    """No segment name appears as (or inside) any text value of the input tables."""
    for frame in (transactions, engagement):
        for column in frame.columns:
            values = set(frame[column].astype(str).unique())
            for name in segment_names:
                assert not any(name in value for value in values), (column, name)


# ---------- save_data ----------

def test_save_data_writes_three_csvs(tmp_path):
    """save_data creates the folder and writes the 3 CSVs with all rows."""
    tx, eng, truth = generate_data(n_customers=200, seed=5)
    target = tmp_path / "new_data_dir"
    save_data(tx, eng, truth, data_dir=target)
    for name, frame in [("transactions.csv", tx), ("engagement.csv", eng),
                        ("ground_truth.csv", truth)]:
        path = target / name
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == len(frame)
        assert list(loaded.columns) == list(frame.columns)
