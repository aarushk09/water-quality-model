"""Publication matplotlib style migrated from model.py."""

from __future__ import annotations

import matplotlib.pyplot as plt

PUBLICATION_RC = {
    "font.size": 12,
    "font.family": "serif",
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.major.size": 6,
    "ytick.minor.size": 3,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.fancybox": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]


def apply_publication_style() -> None:
    """Apply serif/300dpi publication defaults."""
    plt.rcParams.update(PUBLICATION_RC)
