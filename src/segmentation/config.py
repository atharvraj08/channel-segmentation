"""Project-wide settings: paths, seed, data size and the k range to search.

Everything that more than one module needs lives here so there is a single
place to change it.
"""

from pathlib import Path

# config.py lives in <root>/src/segmentation/, so the root is two levels up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
CHARTS_DIR: Path = OUTPUT_DIR / "charts"

# One seed drives everything so results are reproducible.
SEED: int = 42
N_CUSTOMERS: int = 5000
N_TRUE_SEGMENTS: int = 5
K_RANGE: range = range(2, 9)  # k = 2..8 inclusive

CATEGORIES: list[str] = ["electronics", "apparel", "grocery", "home", "beauty"]
N_DAYS: int = 365  # length of the transaction history
# Fixed end date (instead of "today") so generated data never changes between runs.
END_DATE: str = "2026-09-30"
