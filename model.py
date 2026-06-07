"""
DEPRECATED: Legacy TensorFlow LSTM baseline.

Use the PyTorch pipeline instead:

    python train.py --config configs/default.yaml

For NWIS multi-site data:

    python -m data.nwis_fetch --config configs/chattahoochee_graph.yaml
"""

from __future__ import annotations

from pathlib import Path
import warnings

warnings.warn(
    "model.py is deprecated. Use train.py with configs/default.yaml.",
    DeprecationWarning,
    stacklevel=2,
)


def run_legacy_baseline():
    """Optional entry to run the original TensorFlow script if installed."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "legacy_model", __file__ + ".legacy"
        )
    except Exception:
        pass
    raise RuntimeError(
        "Legacy TensorFlow baseline removed. See README.md for PyTorch training."
    )


if __name__ == "__main__":
    print(__doc__)
    import subprocess
    import sys

    subprocess.run([sys.executable, str(Path(__file__).parent / "train.py")] + sys.argv[1:])
