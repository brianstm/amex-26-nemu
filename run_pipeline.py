"""Run the full NEMU pipeline: generate → notice → explain → match → uplift."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_working_scipy() -> None:
    """Fail fast if SciPy cannot load (Homebrew Python 3.10 on recent macOS)."""
    try:
        from scipy import sparse  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "SciPy failed to import from:\n"
            f"  {sys.executable}\n"
            f"  {exc}\n\n"
            "Do not use Homebrew python3.10 for this repo (its SciPy binary is "
            "broken on this macOS, and pip-installing there conflicts with "
            "tensorflow/mediapipe).\n\n"
            "Use the project venv:\n"
            "  python3.12 -m venv .venv\n"
            "  .venv/bin/pip install -r requirements.txt\n"
            "  .venv/bin/python run_pipeline.py\n"
        ) from exc


_require_working_scipy()

from data.generate_synthetic import generate
from explain.cause_classifier import run as run_explain
from match.behavioral_segments import run as run_segments
from match.discover_merchants import run as run_discovery
from match.merchant_clustering import run as run_merchants
from match.merchant_targets_detail import run as run_merchant_detail
from match.uplift_model import run as run_uplift
from notice.counterfactual_model import run as run_notice
from uplift.holdout_measurement import run as run_holdout


def main() -> None:
    generate()
    run_notice()
    run_explain()
    run_merchants()
    run_merchant_detail()
    run_discovery()
    run_uplift()
    run_segments()
    run_holdout()
    print("\nPipeline complete. Launch the demo with:")
    print("  .venv/bin/streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
