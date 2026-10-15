"""End-to-end tests for run_all.py (acceptance criterion A12) and reproducibility."""

import json
import subprocess
import sys
import time

import pandas as pd

from segmentation import config

import run_all

CHART_NAMES = ["silhouette_by_k.png", "pca_segments.png",
               "segment_profiles.png", "channel_mix.png"]
OUTPUT_FILES = ["metrics.json", "segment_profiles.csv", "customer_segments.csv"]
DATA_FILES = ["transactions.csv", "engagement.csv", "ground_truth.csv"]


def assert_non_empty_file(path):
    """Fail with a clear message if path is missing or empty."""
    assert path.is_file(), f"missing {path}"
    assert path.stat().st_size > 0, f"empty {path}"


def test_run_all_script_end_to_end():
    """`python run_all.py` exits 0 in < 120 s and writes every output file."""
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "run_all.py"], cwd=config.PROJECT_ROOT,
        capture_output=True, text=True, timeout=120,
    )
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, result.stderr[-2000:]
    assert elapsed < 120

    output_dir = config.PROJECT_ROOT / "outputs"
    for name in OUTPUT_FILES:
        assert_non_empty_file(output_dir / name)
    for name in CHART_NAMES:
        assert_non_empty_file(output_dir / "charts" / name)
    for name in DATA_FILES:
        assert_non_empty_file(config.PROJECT_ROOT / "data" / name)

    # The k reported in metrics must match what was actually written.
    metrics = json.loads((output_dir / "metrics.json").read_text())
    segments = pd.read_csv(output_dir / "customer_segments.csv")
    assert metrics["chosen_k"] == segments["segment"].nunique()
    assert len(segments) == config.N_CUSTOMERS


def test_pipeline_writes_charts_into_output_dir(pipeline_run):
    """run_pipeline puts the 4 charts in output_dir / "charts" (not the default folder)."""
    for name in CHART_NAMES:
        assert_non_empty_file(pipeline_run["output_dir"] / "charts" / name)
    for name in OUTPUT_FILES:
        assert_non_empty_file(pipeline_run["output_dir"] / name)
    for name in DATA_FILES:
        assert_non_empty_file(pipeline_run["data_dir"] / name)


def test_pipeline_is_reproducible(pipeline_run, tmp_path):
    """A second run with the same seed gives identical metrics (except runtime)."""
    second = run_all.run_pipeline(data_dir=tmp_path / "data", output_dir=tmp_path / "outputs",
                                  make_charts=False)
    first = dict(pipeline_run["metrics"])
    first.pop("runtime_seconds")
    second = dict(second)
    second.pop("runtime_seconds")
    assert first == second

    # The written segment files must match row for row too.
    seg_a = pd.read_csv(pipeline_run["output_dir"] / "customer_segments.csv")
    seg_b = pd.read_csv(tmp_path / "outputs" / "customer_segments.csv")
    pd.testing.assert_frame_equal(seg_a, seg_b)
