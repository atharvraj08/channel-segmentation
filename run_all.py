"""Run the whole customer-segmentation pipeline end to end.

Usage (from the project root):
    .venv/bin/python run_all.py

Steps: generate data -> build features -> scale -> choose k -> cluster ->
evaluate -> save CSV/JSON -> draw charts.
"""

import sys
import time
from pathlib import Path

# Put src/ on the path ourselves so the script works without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from segmentation import cluster, config, evaluate, features, generate_data, visualize  # noqa: E402


def _build_metrics(seed, n_customers, best_k, scores, X, labels, segments, profiles,
                   data_dir) -> dict:
    """Collect the metrics.json fields (without runtime) into one dict."""
    return {
        "seed": seed,
        "n_customers": n_customers,
        "chosen_k": int(best_k),
        # JSON keys must be strings, so convert the int keys here.
        "silhouette_by_k": {str(k): float(v) for k, v in scores.items()},
        "silhouette": evaluate.compute_silhouette(X, labels),
        # Ground truth is read only now, after the segments are final: it scores, never steers.
        "ari": evaluate.compute_ari(segments, evaluate.load_ground_truth(data_dir)),
        "segment_sizes": {str(s): int(n) for s, n in profiles["size"].items()},
        "segment_names": {str(s): str(n) for s, n in profiles["name"].items()},
    }


def run_pipeline(seed: int = config.SEED, n_customers: int = config.N_CUSTOMERS,
                 data_dir: Path = config.DATA_DIR, output_dir: Path = config.OUTPUT_DIR,
                 make_charts: bool = True) -> dict:
    """Run every step and return the metrics dict (also saved as metrics.json)."""
    start = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)

    # 1. Data: generate and save, then read back from CSV like a real pipeline would.
    transactions, engagement, truth = generate_data.generate_data(n_customers=n_customers, seed=seed)
    generate_data.save_data(transactions, engagement, truth, data_dir=data_dir)
    transactions, engagement = features.load_raw_data(data_dir)

    # 2. Features and clustering.
    frame = features.build_features(transactions, engagement)
    X, _ = features.scale_features(frame)
    best_k, scores = cluster.choose_k(X, seed=seed)
    labels = cluster.fit_segments(X, best_k, seed=seed)
    segments = cluster.save_segments(frame.index, labels, output_dir=output_dir)

    # 3. Evaluation and saved results.
    profiles = evaluate.build_profiles(frame, labels)
    evaluate.save_profiles(profiles, output_dir=output_dir)
    metrics = _build_metrics(seed, n_customers, best_k, scores, X, labels, segments,
                             profiles, data_dir)

    # 4. Charts (optional so tests can skip the slowest, purely visual step).
    if make_charts:
        visualize.make_all_charts(X, labels, scores, profiles, charts_dir=output_dir / "charts")

    metrics["runtime_seconds"] = round(time.perf_counter() - start, 3)
    evaluate.save_metrics(metrics, output_dir=output_dir)
    return metrics


def print_summary(metrics: dict) -> None:
    """Print a short, human-readable summary of a pipeline run."""
    print("Customer segmentation finished")
    print(f"  customers : {metrics['n_customers']}  (seed {metrics['seed']})")
    print(f"  chosen k  : {metrics['chosen_k']}")
    print(f"  silhouette: {metrics['silhouette']:.3f}")
    print(f"  ARI       : {metrics['ari']:.3f}  (vs hidden ground truth)")
    print("  segments  :")
    for segment, name in metrics["segment_names"].items():
        print(f"    {segment}: {name:<30} {metrics['segment_sizes'][segment]:>5} customers")
    print(f"  runtime   : {metrics['runtime_seconds']:.1f} s")
    print(f"  outputs   : {config.OUTPUT_DIR}  (charts in {config.CHARTS_DIR})")


def main() -> int:
    """Run run_pipeline() with defaults, print a short summary, return 0."""
    print_summary(run_pipeline())
    return 0


if __name__ == "__main__":
    sys.exit(main())
