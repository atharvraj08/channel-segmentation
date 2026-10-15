"""Turn raw transactions + engagement into one row of numeric features per customer.

The model code never sees the hidden answer key: this module only reads
transactions.csv and engagement.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from segmentation import config

FEATURE_COLUMNS: list[str] = [
    "total_spend", "order_count", "avg_basket", "online_share",
    "online_orders", "store_orders", "discount_rate", "recency_days",
    "n_categories", "top_category_share", "grocery_share",
    "email_open_rate", "app_sessions", "web_sessions", "loyalty_member",
]

# Money and count columns are heavily right-skewed (a few huge spenders).
# log1p squashes that tail so a handful of extreme customers do not dominate
# the distances KMeans relies on. log1p (not log) keeps zeros at zero.
LOG_COLUMNS: list[str] = [
    "total_spend", "order_count", "avg_basket", "online_orders", "store_orders",
    "recency_days", "app_sessions", "web_sessions",
]

# Features actually used for clustering. All of them: each one describes a
# real behaviour (channel, value, frequency, deals, category, engagement), and
# using everything avoids hand-picking features to fit a desired answer.
MODEL_FEATURES: list[str] = list(FEATURE_COLUMNS)


def load_raw_data(data_dir: Path = config.DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, engagement) read from CSV in data_dir.

    Only these two files are read; the answer key file is never touched.
    """
    data_dir = Path(data_dir)
    transactions = pd.read_csv(data_dir / "transactions.csv")
    engagement = pd.read_csv(data_dir / "engagement.csv")
    return transactions, engagement


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide element-wise, turning any 0/0 (or x/0) into 0 instead of NaN/inf."""
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(0.0)


def _purchase_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Spend, order counts, channel mix, discount use and recency per customer."""
    tx = transactions.copy()
    is_online = tx["channel"] == "online"
    tx["online_amount"] = tx["amount"].where(is_online, 0.0)
    tx["is_online"] = is_online.astype(float)
    tx["is_store"] = (~is_online).astype(float)
    # CSVs may give "True"/"False" strings or bools; normalise to 1.0/0.0.
    tx["discount"] = tx["used_discount"].astype(str).str.lower().eq("true").astype(float)
    tx["date"] = pd.to_datetime(tx["date"])

    grouped = tx.groupby("customer_id")
    out = pd.DataFrame({
        "total_spend": grouped["amount"].sum(),
        "order_count": grouped.size().astype(float),
        "online_orders": grouped["is_online"].sum(),
        "store_orders": grouped["is_store"].sum(),
        "discount_rate": grouped["discount"].mean(),
    })
    out["avg_basket"] = _safe_divide(out["total_spend"], out["order_count"])
    out["online_share"] = _safe_divide(grouped["online_amount"].sum(), out["total_spend"])
    # Recency is measured against the latest date in the whole table, so it
    # does not depend on when the code is run.
    out["recency_days"] = (tx["date"].max() - grouped["date"].max()).dt.days.astype(float)
    return out


def _category_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Category breadth and concentration (shares of spend) per customer."""
    spend = transactions.pivot_table(index="customer_id", columns="category",
                                     values="amount", aggfunc="sum", fill_value=0.0)
    total = spend.sum(axis=1)
    grocery = spend["grocery"] if "grocery" in spend.columns else pd.Series(0.0, index=spend.index)
    return pd.DataFrame({
        "n_categories": (spend > 0).sum(axis=1).astype(float),
        "top_category_share": _safe_divide(spend.max(axis=1), total),
        "grocery_share": _safe_divide(grocery, total),
    })


def _engagement_features(engagement: pd.DataFrame) -> pd.DataFrame:
    """Copy the digital engagement columns, renamed, with loyalty as 1.0/0.0."""
    eng = engagement.set_index("customer_id")
    loyalty = eng["loyalty_member"].astype(str).str.lower().eq("true").astype(float)
    return pd.DataFrame({
        "email_open_rate": eng["email_open_rate"].astype(float),
        "app_sessions": eng["app_sessions_per_month"].astype(float),
        "web_sessions": eng["web_sessions_per_month"].astype(float),
        "loyalty_member": loyalty,
    })


def build_features(transactions: pd.DataFrame, engagement: pd.DataFrame) -> pd.DataFrame:
    """One row per customer in `engagement` (index = sorted customer_id),
    columns = FEATURE_COLUMNS in order, raw unscaled float values, no NaN/inf."""
    customers = sorted(engagement["customer_id"])
    parts = [
        _purchase_features(transactions),
        _category_features(transactions),
        _engagement_features(engagement),
    ]
    # Reindex on the engagement customer list so customers without purchases
    # still get a row (filled with 0) and stray ids are dropped.
    frame = pd.concat(parts, axis=1).reindex(customers).fillna(0.0)
    frame.index.name = "customer_id"
    frame = frame[FEATURE_COLUMNS].astype(float)
    return frame.replace([np.inf, -np.inf], 0.0)


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Take df[MODEL_FEATURES], log1p the LOG_COLUMNS among them, then standardize.

    Returns (X, scaler); rows of X keep the order of df. Standardizing puts every
    feature on the same footing so no single unit (e.g. dollars) dominates.
    """
    values = df[MODEL_FEATURES].astype(float).copy()
    for column in MODEL_FEATURES:
        if column in LOG_COLUMNS:
            values[column] = np.log1p(values[column])
    scaler = StandardScaler()
    X = scaler.fit_transform(values.to_numpy())
    return X, scaler
