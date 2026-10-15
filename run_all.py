# TEMPORARY stub by model_agent, to be replaced by viz_agent
"""Temporary pipeline runner so the Model Agent can run the evaluation tests."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from segmentation import cluster, config, evaluate, features, generate_data  # noqa: E402


def run_pipeline(seed: int = config.SEED, n_customers: int = config.N_CUSTOMERS,
                 data_dir: Path = config.DATA_DIR, output_dir: Path = config.OUTPUT_DIR,
                 make_charts: bool = True) -> dict:
    """Run data -> features -> clustering -> evaluation; return metrics (charts ignored)."""
    start = time.time()
    tx, eng, truth = generate_data.generate_data(n_customers=n_customers, seed=seed)
    generate_data.save_data(tx, eng, truth, data_dir=data_dir)
    tx, eng = features.load_raw_data(data_dir)
    frame = features.build_features(tx, eng)
    X, _ = features.scale_features(frame)
    best_k, scores = cluster.choose_k(X, seed=seed)
    labels = cluster.fit_segments(X, best_k, seed=seed)
    segments = cluster.save_segments(frame.index, labels, output_dir=output_dir)
    profiles = evaluate.build_profiles(frame, labels)
    evaluate.save_profiles(profiles, output_dir=output_dir)
    metrics = {
        "seed": seed,
        "n_customers": n_customers,
        "chosen_k": int(best_k),
        "silhouette_by_k": {str(k): v for k, v in scores.items()},
        "silhouette": evaluate.compute_silhouette(X, labels),
        "ari": evaluate.compute_ari(segments, evaluate.load_ground_truth(data_dir)),
        "segment_sizes": {str(s): int(n) for s, n in profiles["size"].items()},
        "segment_names": {str(s): n for s, n in profiles["name"].items()},
        "runtime_seconds": round(time.time() - start, 3),
    }
    evaluate.save_metrics(metrics, output_dir=output_dir)
    return metrics


def main() -> int:
    """Run the pipeline with defaults and print a summary."""
    print(run_pipeline())
    return 0


if __name__ == "__main__":
    sys.exit(main())
