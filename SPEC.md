# SPEC: Omnichannel Customer Segmentation

This is the contract every agent follows. Function names, arguments, return types,
column names and file names below are **exact**. Do not rename anything.

Environment: Python 3.14 in `.venv`. The package lives in `src/segmentation` and is
importable as `segmentation` (pytest is configured in `pyproject.toml` with
`pythonpath = ["src", "."]`, so `import segmentation` and `import run_all` both work in tests).
Allowed libraries: numpy, pandas, scikit-learn, matplotlib, pytest (stdlib is fine).

---

## `src/segmentation/config.py` (Data Agent)

```python
from pathlib import Path
PROJECT_ROOT: Path        # repository root (parent of src/)
DATA_DIR: Path            # PROJECT_ROOT / "data"
OUTPUT_DIR: Path          # PROJECT_ROOT / "outputs"
CHARTS_DIR: Path          # OUTPUT_DIR / "charts"
SEED: int = 42
N_CUSTOMERS: int = 5000
N_TRUE_SEGMENTS: int = 5
K_RANGE: range = range(2, 9)          # k = 2..8 inclusive
CATEGORIES: list[str] = ["electronics", "apparel", "grocery", "home", "beauty"]
N_DAYS: int = 365                     # length of the transaction history
END_DATE: str = "2026-09-30"          # last possible transaction date (fixed so output is reproducible)
```

## `src/segmentation/generate_data.py` (Data Agent)

```python
def generate_data(n_customers: int = config.N_CUSTOMERS,
                  seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (transactions, engagement, ground_truth)."""

def save_data(transactions: pd.DataFrame, engagement: pd.DataFrame,
              ground_truth: pd.DataFrame, data_dir: Path = config.DATA_DIR) -> None:
    """Write transactions.csv, engagement.csv, ground_truth.csv into data_dir (create dir if needed)."""

def main() -> None:
    """Generate with config defaults and save to config.DATA_DIR."""
```

Exact columns:

- `transactions`: `customer_id` (str, e.g. `"C00001"`), `date` (str `YYYY-MM-DD`, within the
  365 days ending `END_DATE`), `channel` (`"online"` or `"store"`), `amount` (float > 0, rounded to 2 dp),
  `category` (one of `CATEGORIES`), `used_discount` (bool).
- `engagement` (exactly one row per customer): `customer_id`, `email_open_rate` (float in [0, 1]),
  `app_sessions_per_month` (float ≥ 0), `web_sessions_per_month` (float ≥ 0),
  `loyalty_member` (bool), `days_since_last_visit` (int, 0..365).
- `ground_truth`: `customer_id`, `true_segment` (str, one of the 5 segment names below).

Rules:
- Exactly `n_customers` customers; **every customer has at least 1 transaction**.
- Customer ids are `"C" + 5-digit zero-padded number` starting at `C00001`.
- 5 planted segments: `"Digital Deal-Seekers"`, `"In-Store Loyalists"`, `"Omnichannel VIPs"`,
  `"Occasional Browsers"`, `"Convenience Grocery Shoppers"`. Unequal sizes, each ≥ 8% of customers.
- Realistic noise so segments overlap somewhat, but an honest KMeans on good features must be
  able to reach ARI ≥ 0.70 against the truth.
- Fully determined by `seed` (use one `np.random.default_rng(seed)`); different seeds give different data.
- `transactions` and `engagement` must NOT contain `true_segment` or any segment name.
- Sorted deterministically (transactions by customer_id then date); no missing values.

## `src/segmentation/features.py` (Model Agent)

```python
FEATURE_COLUMNS: list[str] = [
    "total_spend", "order_count", "avg_basket", "online_share",
    "online_orders", "store_orders", "discount_rate", "recency_days",
    "n_categories", "top_category_share", "grocery_share",
    "email_open_rate", "app_sessions", "web_sessions", "loyalty_member",
]
LOG_COLUMNS: list[str]     # skewed money/count columns that get np.log1p before scaling
MODEL_FEATURES: list[str]  # subset of FEATURE_COLUMNS actually used for clustering (default: all)

def load_raw_data(data_dir: Path = config.DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions, engagement) read from CSV. Never reads ground_truth.csv."""

def build_features(transactions: pd.DataFrame, engagement: pd.DataFrame) -> pd.DataFrame:
    """One row per customer in `engagement`, index = customer_id (sorted), columns = FEATURE_COLUMNS
    in that order, raw (unscaled) values, all floats, no NaN/inf."""

def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Take df[MODEL_FEATURES], apply np.log1p to the LOG_COLUMNS among them, then fit a
    StandardScaler. Returns (X, scaler). X has shape (len(df), len(MODEL_FEATURES)),
    rows in the same order as df, every column mean ≈ 0 and std ≈ 1."""
```

Definitions: `online_share` = online spend / total spend (0..1); `discount_rate` = share of orders
with a discount (0..1); `recency_days` = days from the customer's last purchase to the latest date in
the whole transactions table; `top_category_share` / `grocery_share` = share of spend (0..1);
`loyalty_member` = 1.0/0.0; `app_sessions`, `web_sessions`, `email_open_rate` copied from engagement.
Customers with zero orders in a channel get 0 for that channel; any 0/0 becomes 0.

## `src/segmentation/cluster.py` (Model Agent)

```python
def choose_k(X: np.ndarray, k_range=config.K_RANGE,
             seed: int = config.SEED) -> tuple[int, dict[int, float]]:
    """Fit a model for every k in k_range, return (best_k, {k: silhouette_score})."""

def fit_segments(X: np.ndarray, k: int, seed: int = config.SEED) -> np.ndarray:
    """Return integer labels 0..k-1, shape (len(X),). Deterministic for a given seed."""

def save_segments(customer_ids, labels: np.ndarray,
                  output_dir: Path = config.OUTPUT_DIR) -> pd.DataFrame:
    """Write output_dir/customer_segments.csv with columns customer_id, segment; return that frame."""
```

KMeans with `n_init=10, random_state=seed` (GaussianMixture allowed if it scores better on
silhouette; document in DECISIONS.md). Never reads `ground_truth.csv`; the strings
`ground_truth` and `true_segment` must not appear anywhere in `features.py` or `cluster.py`.

## `src/segmentation/evaluate.py` (Model Agent)

```python
PROFILE_COLUMNS: list[str]   # key features shown in profiles; must include "online_share",
                             # "total_spend", "order_count", "discount_rate", "grocery_share"

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float
def load_ground_truth(data_dir: Path = config.DATA_DIR) -> pd.DataFrame   # the ONLY reader of ground_truth.csv
def compute_ari(segments: pd.DataFrame, ground_truth: pd.DataFrame) -> float
    """segments has customer_id, segment; merged with ground_truth on customer_id."""
def build_profiles(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame
    """Index = segment (int). Columns = PROFILE_COLUMNS (raw-feature means) + "size" (int count)
    + "share" (fraction of customers) + "name" (str from name_segments)."""
def name_segments(profiles: pd.DataFrame) -> dict[int, str]
    """Simple documented if/else rules on profile means; every segment gets a non-empty name."""
def save_metrics(metrics: dict, output_dir: Path = config.OUTPUT_DIR) -> None
    """Write output_dir/metrics.json (indent=2)."""
def save_profiles(profiles: pd.DataFrame, output_dir: Path = config.OUTPUT_DIR) -> None
    """Write output_dir/segment_profiles.csv."""
```

`metrics.json` keys (exact): `seed`, `n_customers`, `chosen_k`, `silhouette_by_k`
(dict with string keys "2".."8"), `silhouette`, `ari`, `segment_sizes` (dict "segment id" -> int),
`segment_names` (dict "segment id" -> str), `runtime_seconds`.

## `src/segmentation/visualize.py` (Viz & Pipeline Agent)

```python
def plot_silhouette_by_k(scores: dict[int, float], path: Path) -> None
def plot_pca_segments(X: np.ndarray, labels: np.ndarray, path: Path, seed: int = config.SEED) -> None
def plot_segment_profiles(profiles: pd.DataFrame, path: Path) -> None     # heatmap of standardized means
def plot_channel_mix(profiles: pd.DataFrame, path: Path) -> None          # online vs store share per segment
def make_all_charts(X, labels, scores, profiles, charts_dir: Path = config.CHARTS_DIR) -> list[Path]
    """Create charts_dir and write silhouette_by_k.png, pca_segments.png,
    segment_profiles.png, channel_mix.png. Return the 4 paths."""
```
Use the non-interactive backend (`matplotlib.use("Agg")`). Close every figure.

## `run_all.py` (Viz & Pipeline Agent), at the project root

```python
def run_pipeline(seed: int = config.SEED, n_customers: int = config.N_CUSTOMERS,
                 data_dir: Path = config.DATA_DIR, output_dir: Path = config.OUTPUT_DIR,
                 make_charts: bool = True) -> dict:
    """generate+save data -> load_raw_data -> build_features -> scale_features -> choose_k ->
    fit_segments -> save_segments -> silhouette, ARI, profiles, names -> save_profiles,
    save_metrics -> charts (into output_dir / "charts"). Returns the metrics dict."""

def main() -> int:
    """Run run_pipeline() with defaults, print a short summary, return 0."""

if __name__ == "__main__":
    sys.exit(main())
```
`run_all.py` must put `src/` on `sys.path` itself so `.venv/bin/python run_all.py` works from the
project root. Outputs: `outputs/metrics.json`, `outputs/segment_profiles.csv`,
`outputs/customer_segments.csv`, `outputs/charts/{silhouette_by_k,pca_segments,segment_profiles,channel_mix}.png`.

---

## Acceptance criteria

| # | Requirement |
|---|---|
| A1 | ~5,000 customers, required columns, no missing values, amounts > 0, rates in [0, 1] |
| A2 | Same seed twice → identical data |
| A3 | Ground-truth labels absent from transactions and engagement |
| A4 | Feature matrix one row per customer, no NaN/inf, scaled columns mean ≈ 0, std ≈ 1 |
| A5 | `features.py` and `cluster.py` never contain `ground_truth` or `true_segment` |
| A6 | Chosen k is 5 ± 1 |
| A7 | ARI vs hidden truth ≥ 0.70 |
| A8 | Silhouette of chosen model ≥ 0.15 |
| A9 | Every predicted segment ≥ 3% of customers |
| A10 | Re-clustering with a different seed → ARI ≥ 0.80 against the first run |
| A11 | Some segment online_share > 0.65 and some segment online_share < 0.35 |
| A12 | `run_all.py` < 120 s, exit 0, creates every output file |
| A13 | Full `pytest -q` passes in < 60 s total |
| A14 | Tests are meaningful (they must catch: shuffled feature rows before clustering, hard-coded k=2, removed scaling, generator ignoring the seed) |

Rules for everyone: fixed seeds everywhere, docstring on every function, short functions,
comments explaining *why*, beginner-readable. Log judgment calls in `DECISIONS.md`
(append only, prefix each entry with your agent name).
