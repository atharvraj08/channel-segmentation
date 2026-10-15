"""Synthetic omnichannel retail data with 5 planted customer segments.

How it works, in short:
1. Every customer is assigned a hidden segment (unequal sizes).
2. Each customer draws their *own* behaviour parameters (online probability,
   order count, basket size, discount habit, category mix, engagement) from
   distributions that depend on the segment. Because these are random draws,
   customers from different segments overlap a little, like real people.
3. Individual transactions are then sampled from each customer's parameters.

Only `ground_truth` knows the segment; `transactions` and `engagement` never do.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from segmentation import config

# Per-segment recipe. Read each row as "a typical customer in this segment...".
#   share          fraction of all customers
#   orders         mean orders per year (negative binomial), with dispersion `orders_r`
#                  (smaller r = more spread between customers)
#   online_ab      Beta(a, b) for the customer's probability that an order is online
#   basket         median order amount in dollars (lognormal)
#   discount_ab    Beta(a, b) for the probability an order uses a discount
#   category_alpha Dirichlet weights over CATEGORIES (electronics, apparel, grocery, home, beauty)
#   email_ab       Beta(a, b) for email open rate
#   app, web       mean sessions per month (gamma distributed)
#   loyalty_p      probability of being a loyalty member
#   gap_days       mean days since the last purchase (exponential)
SEGMENTS: dict[str, dict] = {
    "Digital Deal-Seekers": dict(
        share=0.24, orders=26, orders_r=3, online_ab=(8, 1.6), basket=28,
        discount_ab=(6, 3), category_alpha=(3, 4, 1, 2, 3),
        email_ab=(3, 2), app=14, web=10, loyalty_p=0.30, gap_days=10,
    ),
    "In-Store Loyalists": dict(
        share=0.22, orders=16, orders_r=4, online_ab=(1.3, 7), basket=75,
        discount_ab=(1.5, 12), category_alpha=(2, 3, 2, 3, 2),
        email_ab=(3, 6), app=1.5, web=2, loyalty_p=0.85, gap_days=18,
    ),
    "Omnichannel VIPs": dict(
        share=0.12, orders=50, orders_r=4, online_ab=(5, 5), basket=220,
        discount_ab=(2, 8), category_alpha=(4, 4, 2, 4, 3),
        email_ab=(4, 1.8), app=12, web=8, loyalty_p=0.85, gap_days=6,
    ),
    "Occasional Browsers": dict(
        share=0.24, orders=2.5, orders_r=4, online_ab=(4, 3), basket=55,
        discount_ab=(2, 5), category_alpha=(3, 3, 1, 2, 2),
        email_ab=(1.5, 8), app=1.5, web=12, loyalty_p=0.10, gap_days=60,
    ),
    "Convenience Grocery Shoppers": dict(
        share=0.18, orders=60, orders_r=4, online_ab=(1.2, 6), basket=30,
        discount_ab=(2, 6), category_alpha=(0.4, 0.6, 12, 1.2, 0.8),
        email_ab=(3, 5), app=3, web=2, loyalty_p=0.60, gap_days=4,
    ),
}

# How much a single order's amount varies around the customer's own typical basket.
ORDER_AMOUNT_SIGMA = 0.45
# How much customers' typical baskets vary within a segment.
CUSTOMER_BASKET_SIGMA = 0.5


def _assign_segments(n_customers: int, rng: np.random.Generator) -> np.ndarray:
    """Return an array of segment indices (0..4) with exactly the planned sizes.

    Using exact counts (instead of random draws) guarantees every segment
    keeps its planned share, so none can fall below 8% by bad luck.
    """
    shares = np.array([s["share"] for s in SEGMENTS.values()])
    counts = np.floor(shares * n_customers).astype(int)
    counts[0] += n_customers - counts.sum()  # give rounding leftovers to segment 0
    seg_idx = np.repeat(np.arange(len(SEGMENTS)), counts)
    return rng.permutation(seg_idx)  # shuffle so ids are not grouped by segment


def _param(name: str, seg_idx: np.ndarray) -> np.ndarray:
    """Look up a per-segment parameter for every customer (vectorised)."""
    values = np.array([s[name] for s in SEGMENTS.values()], dtype=float)
    return values[seg_idx]


def _draw_customer_params(seg_idx: np.ndarray, rng: np.random.Generator) -> dict:
    """Draw each customer's personal behaviour parameters from their segment's distributions."""
    n = len(seg_idx)
    # Negative binomial = Poisson with gamma-distributed rate -> realistic heavy tail.
    mean_orders, r = _param("orders", seg_idx), _param("orders_r", seg_idx)
    n_orders = rng.negative_binomial(r, r / (r + mean_orders))
    n_orders = np.maximum(n_orders, 1)  # every customer has at least one purchase

    online_ab = _param("online_ab", seg_idx)
    discount_ab = _param("discount_ab", seg_idx)
    email_ab = _param("email_ab", seg_idx)
    alpha = _param("category_alpha", seg_idx)
    # Dirichlet per customer: draw gammas row-wise and normalise (vectorised Dirichlet).
    cat_probs = rng.gamma(alpha)
    cat_probs /= cat_probs.sum(axis=1, keepdims=True)

    return dict(
        n_orders=n_orders,
        p_online=rng.beta(online_ab[:, 0], online_ab[:, 1]),
        p_discount=rng.beta(discount_ab[:, 0], discount_ab[:, 1]),
        basket=_param("basket", seg_idx) * rng.lognormal(0, CUSTOMER_BASKET_SIGMA, n),
        cat_probs=cat_probs,
        email_open_rate=rng.beta(email_ab[:, 0], email_ab[:, 1]),
        # Gamma with shape 2 -> mean = segment mean, moderate spread, never negative.
        app=rng.gamma(2.0, _param("app", seg_idx) / 2.0),
        web=rng.gamma(2.0, _param("web", seg_idx) / 2.0),
        loyalty=rng.random(n) < _param("loyalty_p", seg_idx),
        gap_days=np.minimum(rng.exponential(_param("gap_days", seg_idx)), config.N_DAYS - 30),
    )


def _sample_categories(cat_probs: np.ndarray, owner: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """Pick one category per order from the owning customer's category probabilities.

    Vectorised inverse-CDF sampling: compare one uniform number per order with
    the cumulative probabilities of that order's customer.
    """
    cdf = np.cumsum(cat_probs, axis=1)[owner]
    u = rng.random(len(owner))[:, None]
    idx = (u > cdf).sum(axis=1)
    return np.minimum(idx, len(config.CATEGORIES) - 1)  # guard against float round-off


def _build_transactions(ids: np.ndarray, params: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Expand customer parameters into one row per order."""
    owner = np.repeat(np.arange(len(ids)), params["n_orders"])
    n_tx = len(owner)

    is_online = rng.random(n_tx) < params["p_online"][owner]
    amount = params["basket"][owner] * rng.lognormal(0, ORDER_AMOUNT_SIGMA, n_tx)
    amount = np.maximum(np.round(amount, 2), 0.01)  # strictly positive after rounding
    used_discount = rng.random(n_tx) < params["p_discount"][owner]
    category = np.array(config.CATEGORIES)[_sample_categories(params["cat_probs"], owner, rng)]

    # Orders happen uniformly between the start of history and the customer's last active day.
    # days_ago = 0 means END_DATE. The latest order of each customer is pinned at gap_days so
    # recency reflects the segment's habit.
    gap = np.round(params["gap_days"]).astype(int)
    span = config.N_DAYS - 1 - gap[owner]
    days_ago = gap[owner] + np.floor(rng.random(n_tx) * (span + 1)).astype(int)
    first_of_customer = np.r_[0, np.cumsum(params["n_orders"])[:-1]]
    days_ago[first_of_customer] = gap  # pin one order per customer at the last active day
    dates = pd.Timestamp(config.END_DATE) - pd.to_timedelta(days_ago, unit="D")

    tx = pd.DataFrame({
        "customer_id": ids[owner],
        "date": dates.strftime("%Y-%m-%d"),
        "channel": np.where(is_online, "online", "store"),
        "amount": amount,
        "category": category,
        "used_discount": used_discount,
    })
    # Stable sort keeps the result deterministic even when dates tie.
    return tx.sort_values(["customer_id", "date"], kind="mergesort").reset_index(drop=True)


def _build_engagement(ids: np.ndarray, params: dict, rng: np.random.Generator) -> pd.DataFrame:
    """One row per customer with email/app/web engagement and loyalty status."""
    app = np.round(params["app"], 2)
    web = np.round(params["web"], 2)
    # A "visit" is a purchase or a digital session. Frequent app/web users have short gaps
    # since their last session, so the visit gap is the smaller of the two.
    session_gap = rng.exponential(30.0 / (app + web + 0.5))
    days_since = np.minimum(params["gap_days"], session_gap)
    return pd.DataFrame({
        "customer_id": ids,
        "email_open_rate": np.round(params["email_open_rate"], 4),
        "app_sessions_per_month": app,
        "web_sessions_per_month": web,
        "loyalty_member": params["loyalty"].astype(bool),
        "days_since_last_visit": np.clip(np.round(days_since), 0, config.N_DAYS).astype(int),
    })


def generate_data(n_customers: int = config.N_CUSTOMERS,
                  seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (transactions, engagement, ground_truth)."""
    rng = np.random.default_rng(seed)  # the single source of randomness
    ids = np.array([f"C{i:05d}" for i in range(1, n_customers + 1)])
    seg_idx = _assign_segments(n_customers, rng)
    params = _draw_customer_params(seg_idx, rng)

    transactions = _build_transactions(ids, params, rng)
    engagement = _build_engagement(ids, params, rng)
    segment_names = np.array(list(SEGMENTS))
    ground_truth = pd.DataFrame({"customer_id": ids, "true_segment": segment_names[seg_idx]})
    return transactions, engagement, ground_truth


def save_data(transactions: pd.DataFrame, engagement: pd.DataFrame,
              ground_truth: pd.DataFrame, data_dir: Path = config.DATA_DIR) -> None:
    """Write transactions.csv, engagement.csv, ground_truth.csv into data_dir (create dir if needed)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(data_dir / "transactions.csv", index=False)
    engagement.to_csv(data_dir / "engagement.csv", index=False)
    ground_truth.to_csv(data_dir / "ground_truth.csv", index=False)


def main() -> None:
    """Generate with config defaults and save to config.DATA_DIR."""
    transactions, engagement, ground_truth = generate_data()
    save_data(transactions, engagement, ground_truth)
    print(f"Saved {len(transactions):,} transactions for {len(engagement):,} customers "
          f"to {config.DATA_DIR}")


if __name__ == "__main__":
    main()
