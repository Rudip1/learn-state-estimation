"""Plotting helpers shared by the notebooks and tools/make_figures.py. No algorithms here."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import _core

#: Colours used consistently through the module.
COLORS = {
    "truth": "#222222",
    "estimate": "#1f77b4",
    "dead_reckoning": "#d62728",
    "measurement": "#2ca02c",
    "landmark": "#ff7f0e",
    "particles": "#9467bd",
}


def new_axes(ax=None, equal: bool = True, figsize=(5, 5)):
    """Return ``ax`` or a fresh axes, optionally with equal aspect and a light grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if equal:
        ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


def plot_ellipse(ax, mean, cov, p: float = 0.95, **kwargs):
    """Draw the confidence ellipse of a 2-D Gaussian (eq. 1.17) on ``ax``."""
    pts = _core.ellipse_points(np.asarray(mean, float)[:2], np.asarray(cov, float)[:2, :2], p, 72)
    kwargs.setdefault("lw", 1.2)
    return ax.plot(pts[:, 0], pts[:, 1], **kwargs)


def plot_pose(ax, pose, length: float = 0.3, color: str = "k", **kwargs):
    """Draw a planar pose [x, y, theta] as a dot with a heading arrow."""
    x, y, th = pose[:3]
    ax.plot(x, y, "o", color=color, ms=4, **kwargs)
    ax.arrow(x, y, length * np.cos(th), length * np.sin(th), color=color, head_width=0.4 * length,
             length_includes_head=True)


def plot_trajectory(ax, poses, label=None, color=None, **kwargs):
    """Plot the (x, y) columns of an (N, >=2) array of poses."""
    poses = np.asarray(poses)
    return ax.plot(poses[:, 0], poses[:, 1], label=label, color=color, **kwargs)
