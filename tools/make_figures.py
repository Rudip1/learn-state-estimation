#!/usr/bin/env python3
"""Regenerate every figure in 1_theory/figures from the C++ library (through the Python module).

    python tools/make_figures.py          # all chapters
    python tools/make_figures.py 01 05    # only these chapters

Figures are never edited by hand: change this script and re-run it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import state_estimation as se  # noqa: E402
from state_estimation import plotting  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "1_theory" / "figures"
FIGURES: dict[str, list] = {}


def figure(chapter: str):
    def register(fn):
        FIGURES.setdefault(chapter, []).append(fn)
        return fn

    return register


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote 1_theory/figures/{name}")


# ---------------------------------------------------------------- chapter 1
@figure("01")
def gaussian_ellipses():
    rng = se.Rng(3)
    mean = np.array([1.0, 0.5])
    cov = np.array([[1.0, 0.6], [0.6, 0.5]])
    s = se.sample_gaussian(mean, cov, 1500, rng)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    plotting.new_axes(ax)
    ax.plot(s[:, 0], s[:, 1], ".", ms=2, color="0.6", label="samples")
    for k, ls in zip((1, 2, 3), ("-", "--", ":")):
        p = se.chi2_cdf(k * k, 2)
        inside = np.mean([se.mahalanobis_squared(x, mean, cov) <= k * k for x in s])
        plotting.plot_ellipse(ax, mean, cov, p, color=plotting.COLORS["estimate"], ls=ls,
                              label=f"{k}σ: P = {p:.3f}, sampled {inside:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("2-D Gaussian: k-sigma ellipses hold χ²₂ mass")
    save(fig, "01_gaussian_ellipses.png")


@figure("01")
def banana():
    rng = se.Rng(5)
    polar = se.Gaussian(np.array([1.0, 0.0]), np.diag([0.02**2, 0.5**2]))
    s = se.sample_gaussian(polar.mean, polar.cov, 3000, rng)
    xy = np.array([se.polar_to_cartesian(p) for p in s])
    lin = se.propagate_linearized(lambda p: se.polar_to_cartesian(p), polar,
                                  lambda p: se.polar_to_cartesian_jacobian(p))
    mc = se.sample_mean_cov(xy)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    plotting.new_axes(ax)
    ax.plot(xy[:, 0], xy[:, 1], ".", ms=2, color="0.6", label="Monte Carlo samples")
    plotting.plot_ellipse(ax, lin.mean, lin.cov, 0.95, color=plotting.COLORS["dead_reckoning"],
                          label="linearised (95 %)")
    plotting.plot_ellipse(ax, mc.mean, mc.cov, 0.95, color=plotting.COLORS["estimate"],
                          label="sample moments (95 %)")
    ax.plot(*lin.mean, "x", color=plotting.COLORS["dead_reckoning"], ms=9)
    ax.plot(*mc.mean, "+", color=plotting.COLORS["estimate"], ms=11)
    ax.legend(loc="center left", fontsize=8)
    ax.set_title("ρ = 1 m, σ_φ = 0.5 rad through (ρ cos φ, ρ sin φ)")
    save(fig, "01_banana.png")


def main(argv: list[str]) -> int:
    for chapter in sorted(FIGURES):
        if argv and chapter not in argv:
            continue
        for fn in FIGURES[chapter]:
            fn()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
