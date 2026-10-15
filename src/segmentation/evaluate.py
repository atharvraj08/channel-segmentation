"""Score, describe and name the segments, and save the results.

This is the ONLY module allowed to read the hidden answer key
(data/ground_truth.csv), and it uses it purely to *report* ARI after the
segments have already been chosen.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from segmentation import config

# Features shown in the segment profiles: the ones a marketer would read to
# understand a segment (value, frequency, channel, deals, category, engagement).
PROFILE_COLUMNS: list[str] = [
    "total_spend", "order_count", "avg_basket", "online_share", "discount_rate",
    "recency_days", "grocery_share", "email_open_rate", "app_sessions",
    "web_sessions", "loyalty_member",
]

# Thresholds used by name_segments (on raw profile means). Kept here so the
# rules are easy to read and adjust in one place.
GROCERY_HEAVY = 0.5       # more than half of spend is grocery
FEW_ORDERS = 6            # fewer than ~6 orders a year = occasional
VIP_SPEND = 5000          # yearly spend well above every other behaviour
ONLINE_HEAVY = 0.65       # mostly online
STORE_HEAVY = 0.35        # mostly in store
DEAL_HUNTER = 0.4         # discount on more than 40% of orders
LOYAL = 0.5               # most customers in the segment are loyalty members


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean silhouette score of the labelling (-1..1, higher is better)."""
    return float(silhouette_score(X, labels))


def load_ground_truth(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Read data_dir/ground_truth.csv (customer_id, true_segment). Evaluation only."""
    return pd.read_csv(Path(data_dir) / "ground_truth.csv")


def compute_ari(segments: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """Adjusted Rand Index between predicted segments and the true segments.

    The two tables are merged on customer_id so row order never matters.
    ARI ignores which number each segment got; 1 = perfect, ~0 = random.
    """
    merged = segments.merge(ground_truth, on="customer_id", validate="one_to_one")
    return float(adjusted_rand_score(merged["true_segment"], merged["segment"]))


def _name_one(row: pd.Series) -> str:
    """Name one segment from its profile means with simple, ordered if/else rules.

    The first rule that matches wins, so the most distinctive behaviours are
    checked first (a grocery-heavy segment is named for that even if it is
    also store-heavy).
    """
    if row["grocery_share"] > GROCERY_HEAVY:
        return "Convenience Grocery Shoppers"
    if row["order_count"] < FEW_ORDERS:
        return "Occasional Browsers"
    if row["total_spend"] > VIP_SPEND and STORE_HEAVY <= row["online_share"] <= ONLINE_HEAVY:
        return "Omnichannel VIPs"
    if row["online_share"] > ONLINE_HEAVY and row["discount_rate"] > DEAL_HUNTER:
        return "Digital Deal-Seekers"
    if row["online_share"] < STORE_HEAVY and row["loyalty_member"] > LOYAL:
        return "In-Store Loyalists"
    # Fallback: a plain description of the channel mix.
    if row["online_share"] > ONLINE_HEAVY:
        return "Online-Leaning Shoppers"
    if row["online_share"] < STORE_HEAVY:
        return "Store-Leaning Shoppers"
    return "Mixed-Channel Shoppers"


def name_segments(profiles: pd.DataFrame) -> dict[int, str]:
    """Give every segment a short, unique, human-readable name.

    Uses only the PROFILE_COLUMNS means. If two segments get the same name,
    later ones get " 2", " 3", ... appended so names stay unique.
    """
    names: dict[int, str] = {}
    used: dict[str, int] = {}
    for segment, row in profiles.iterrows():
        base = _name_one(row)
        used[base] = used.get(base, 0) + 1
        names[int(segment)] = base if used[base] == 1 else f"{base} {used[base]}"
    return names


def build_profiles(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-segment means of PROFILE_COLUMNS plus size, share and name.

    Index = segment id (int). Means are on the raw (unscaled) features so they
    read in real units (dollars, orders, shares).
    """
    labels = np.asarray(labels).astype(int)
    profiles = features[PROFILE_COLUMNS].groupby(labels).mean()
    profiles.index.name = "segment"
    sizes = pd.Series(labels).value_counts()
    profiles["size"] = [int(sizes[s]) for s in profiles.index]
    profiles["share"] = profiles["size"] / len(labels)
    names = name_segments(profiles)
    profiles["name"] = [names[int(s)] for s in profiles.index]
    return profiles


def save_metrics(metrics: dict, output_dir: Path = config.OUTPUT_DIR) -> None:
    """Write output_dir/metrics.json (indent=2)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def save_profiles(profiles: pd.DataFrame, output_dir: Path = config.OUTPUT_DIR) -> None:
    """Write output_dir/segment_profiles.csv (segment id kept as a column)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output_dir / "segment_profiles.csv")
