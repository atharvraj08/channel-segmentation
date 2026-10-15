"""Shared, session-scoped fixtures.

The expensive steps (generating 5,000 customers, choosing k, running the whole
pipeline) are computed ONCE per test session and shared, so the full suite stays
well under the 60 second budget (acceptance criterion A13).
"""

import pytest

# Project modules are imported inside the fixtures (not at the top) so that a
# missing module only breaks the tests that need it, not the whole session.

SEGMENT_NAMES = [
    "Digital Deal-Seekers",
    "In-Store Loyalists",
    "Omnichannel VIPs",
    "Occasional Browsers",
    "Convenience Grocery Shoppers",
]


@pytest.fixture(scope="session")
def segment_names():
    """The 5 planted segment names from SPEC.md."""
    return list(SEGMENT_NAMES)


@pytest.fixture(scope="session")
def generated():
    """(transactions, engagement, ground_truth) generated in memory with config defaults."""
    from segmentation import generate_data

    return generate_data.generate_data()


@pytest.fixture(scope="session")
def transactions(generated):
    """The generated transactions table."""
    return generated[0]


@pytest.fixture(scope="session")
def engagement(generated):
    """The generated engagement table (one row per customer)."""
    return generated[1]


@pytest.fixture(scope="session")
def ground_truth(generated):
    """The hidden true segments (only tests and evaluate.py may look at these)."""
    return generated[2]


@pytest.fixture(scope="session")
def feature_frame(transactions, engagement):
    """Raw (unscaled) feature frame, one row per customer."""
    from segmentation import features

    return features.build_features(transactions, engagement)


@pytest.fixture(scope="session")
def scaled(feature_frame):
    """(X, scaler) from scale_features on the real feature frame."""
    from segmentation import features

    return features.scale_features(feature_frame)


@pytest.fixture(scope="session")
def X(scaled):
    """The scaled feature matrix used for clustering."""
    return scaled[0]


@pytest.fixture(scope="session")
def k_result(X):
    """(best_k, {k: silhouette}) from choose_k, computed once because it is slow."""
    from segmentation import cluster

    return cluster.choose_k(X)


@pytest.fixture(scope="session")
def labels(X, k_result):
    """Cluster labels for the chosen k with the default seed."""
    from segmentation import cluster, config

    best_k, _ = k_result
    return cluster.fit_segments(X, best_k, seed=config.SEED)


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory):
    """Run the full pipeline once into a temporary folder.

    Returns a dict with the returned metrics plus the data and output folders,
    so tests can inspect the files exactly as the pipeline wrote them.
    """
    import run_all

    base = tmp_path_factory.mktemp("pipeline")
    data_dir = base / "data"
    output_dir = base / "outputs"
    metrics = run_all.run_pipeline(data_dir=data_dir, output_dir=output_dir)
    return {"metrics": metrics, "data_dir": data_dir, "output_dir": output_dir}
