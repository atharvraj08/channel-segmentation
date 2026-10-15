"""Charts that explain the segmentation: how k was chosen, how the segments
separate, what each segment looks like, and how each one splits its spend
between online and store.

Every function writes one PNG and closes its figure, so running the pipeline
many times (e.g. in tests) never leaks memory or opens windows.
"""

from pathlib import Path

import matplotlib

# "Agg" draws straight to files. It must be chosen before pyplot is imported so
# the code works on servers / in tests where there is no screen.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from segmentation import config  # noqa: E402

DPI = 120
# A colour-blind-friendly qualitative palette (Okabe-Ito), one colour per segment.
SEGMENT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                  "#D55E00", "#56B4E9", "#F0E442", "#000000"]
ONLINE_COLOR = "#0072B2"
STORE_COLOR = "#E69F00"


def _segment_color(segment: int) -> str:
    """Return a stable colour for a segment id (wraps around if k > palette size)."""
    return SEGMENT_COLORS[int(segment) % len(SEGMENT_COLORS)]


def _segment_labels(profiles: pd.DataFrame) -> list[str]:
    """Readable labels like "0: Omnichannel VIPs" (just the id if no names exist)."""
    if "name" not in profiles.columns:
        return [str(s) for s in profiles.index]
    return [f"{s}: {name}" for s, name in zip(profiles.index, profiles["name"])]


def _save(fig: plt.Figure, path: Path) -> None:
    """Tighten the layout, write the figure to path and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_silhouette_by_k(scores: dict[int, float], path: Path) -> None:
    """Line chart of silhouette score for each k, with the best k highlighted."""
    ks = sorted(int(k) for k in scores)
    values = [scores[k] for k in ks]
    best_k = max(ks, key=lambda k: scores[k])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, values, marker="o", color="#555555", linewidth=2)
    # A larger coloured marker makes the chosen k obvious at a glance.
    ax.scatter([best_k], [scores[best_k]], s=180, color=ONLINE_COLOR, zorder=3,
               label=f"chosen k = {best_k} ({scores[best_k]:.3f})")
    ax.set_xticks(ks)
    ax.set_xlabel("Number of segments (k)")
    ax.set_ylabel("Silhouette score (higher is better)")
    ax.set_title("Choosing k: silhouette score by number of segments")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    _save(fig, path)


def plot_pca_segments(X: np.ndarray, labels: np.ndarray, path: Path,
                      seed: int = config.SEED, names: dict[int, str] | None = None) -> None:
    """2-D PCA scatter of the scaled features, coloured by predicted segment.

    PCA squeezes the many features into the 2 directions with the most
    variation, which is the best flat picture we can draw of the clusters.
    `names` (segment id -> name) is optional and only used for the legend.
    """
    labels = np.asarray(labels).astype(int)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    for segment in np.unique(labels):
        mask = labels == segment
        label = f"{segment}: {names[segment]}" if names and segment in names else f"Segment {segment}"
        # Small, semi-transparent markers so dense regions stay readable.
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.4,
                   color=_segment_color(segment), label=label)
    ax.set_xlabel(f"PC1 ({explained[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0f}% of variance)")
    ax.set_title("Customer segments in 2-D (PCA of scaled features)")
    legend = ax.legend(loc="best", markerscale=3, fontsize=9)
    for handle in legend.legend_handles:
        handle.set_alpha(1)  # legend dots fully opaque even though points are not
    _save(fig, path)


def plot_segment_profiles(profiles: pd.DataFrame, path: Path) -> None:
    """Heatmap of per-segment feature means, standardized across segments.

    Raw means live on very different scales (dollars vs shares), so each column
    is turned into a z-score across segments: colour shows whether a segment is
    high or low *relative to the other segments*. Cells are annotated with the
    raw mean so the real numbers are still visible.
    """
    numeric = profiles.drop(columns=["name", "size", "share"], errors="ignore")
    numeric = numeric.select_dtypes("number")
    spread = numeric.std(ddof=0).replace(0, 1)  # avoid divide-by-zero on constant columns
    z = (numeric - numeric.mean()) / spread

    fig, ax = plt.subplots(figsize=(1.0 * len(numeric.columns) + 3, 0.7 * len(numeric) + 2.5))
    limit = max(1.0, float(np.abs(z.values).max()))
    image = ax.imshow(z.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(z.shape[0]):
        for col in range(z.shape[1]):
            ax.text(col, row, _format_value(numeric.iat[row, col]), ha="center", va="center",
                    fontsize=8, color="white" if abs(z.iat[row, col]) > 0.6 * limit else "black")
    ax.set_xticks(range(len(numeric.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in numeric.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(numeric)))
    ax.set_yticklabels(_segment_labels(profiles))
    ax.set_title("Segment profiles (colour = z-score across segments, text = raw mean)")
    fig.colorbar(image, ax=ax, label="z-score vs other segments", shrink=0.8)
    _save(fig, path)


def _format_value(value: float) -> str:
    """Short text for a heatmap cell: whole numbers for big values, 2 dp for shares."""
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_channel_mix(profiles: pd.DataFrame, path: Path) -> None:
    """Stacked horizontal bars: share of spend online vs in store, per segment."""
    online = profiles["online_share"].to_numpy(dtype=float)
    store = 1.0 - online
    labels = _segment_labels(profiles)
    rows = np.arange(len(profiles))

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(profiles) + 2))
    ax.barh(rows, online, color=ONLINE_COLOR, label="Online")
    ax.barh(rows, store, left=online, color=STORE_COLOR, label="Store")
    for row, share in zip(rows, online):
        # Percent labels inside each bar part (skipped if the part is too thin).
        if share > 0.08:
            ax.text(share / 2, row, f"{share:.0%}", ha="center", va="center", color="white")
        if 1 - share > 0.08:
            ax.text(share + (1 - share) / 2, row, f"{1 - share:.0%}", ha="center", va="center")
    ax.set_yticks(rows)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # segment 0 at the top, reading order
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of spend")
    ax.set_title("Channel mix: online vs store spend share per segment")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    _save(fig, path)


def make_all_charts(X, labels, scores, profiles,
                    charts_dir: Path = config.CHARTS_DIR) -> list[Path]:
    """Create charts_dir and write the 4 charts. Return their paths."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    paths = [charts_dir / "silhouette_by_k.png", charts_dir / "pca_segments.png",
             charts_dir / "segment_profiles.png", charts_dir / "channel_mix.png"]
    names = profiles["name"].to_dict() if "name" in profiles.columns else None
    plot_silhouette_by_k(scores, paths[0])
    plot_pca_segments(X, labels, paths[1], names=names)
    plot_segment_profiles(profiles, paths[2])
    plot_channel_mix(profiles, paths[3])
    return paths
