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


# ---------------------------------------------------------------- chapter 2
@figure("02")
def odometry_samples():
    o0, o1 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.6, 0.8])
    cases = [("rotation noise", [0.05, 0.001, 0.001, 0.001]),
             ("translation noise", [0.001, 0.001, 0.05, 0.001]),
             ("both", [0.03, 0.01, 0.03, 0.01])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, (title, alpha) in zip(axes, cases):
        rng = se.Rng(2)
        s = np.array([se.sample_odometry_motion(o0, o0, o1, alpha, rng) for _ in range(1500)])
        plotting.new_axes(ax)
        ax.plot(s[:, 0], s[:, 1], ".", ms=2, color=plotting.COLORS["particles"])
        plotting.plot_pose(ax, o0, 0.25)
        plotting.plot_pose(ax, o1, 0.25, color=plotting.COLORS["truth"])
        ax.set_title(f"{title}\nα = {alpha}", fontsize=9)
    save(fig, "02_odometry_samples.png")


@figure("02")
def dead_reckoning():
    dd = se.DifferentialDrive(0.1, 0.5)
    dt, k = 0.1, 1e-4
    u = se.figure_eight_controls(2.0, 200, dt)
    truth = se.integrate_controls(np.zeros(3), u, dt)
    rng = se.Rng(4)
    run = se.run_dead_reckoning(np.zeros(3), np.zeros((3, 3)), se.simulate_wheel_travel(u, dt, dd, k, rng), dd, k)
    ends = []
    for _ in range(300):
        r = se.run_dead_reckoning(np.zeros(3), np.zeros((3, 3)), se.simulate_wheel_travel(u, dt, dd, k, rng), dd, k)
        ends.append(r.poses[-1])
    ends = np.array(ends)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plotting.new_axes(ax)
    plotting.plot_trajectory(ax, truth, "ground truth", plotting.COLORS["truth"])
    plotting.plot_trajectory(ax, run.poses, "dead reckoning", plotting.COLORS["dead_reckoning"])
    for i in range(0, len(run.poses), 50):
        plotting.plot_ellipse(ax, run.poses[i], run.covariances[i], 0.95, color=plotting.COLORS["estimate"])
    ax.plot(ends[:, 0], ends[:, 1], ".", ms=3, color="0.5", label="final pose, 300 runs")
    plotting.plot_ellipse(ax, run.poses[-1], run.covariances[-1], 0.95, color=plotting.COLORS["estimate"],
                          label="predicted 95 % ellipse")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title("Dead reckoning on a figure eight: uncertainty grows without bound")
    save(fig, "02_dead_reckoning.png")


# ---------------------------------------------------------------- chapter 3
@figure("03")
def likelihoods():
    truth = np.array([2.0, 1.0, 0.6])
    m = np.array([5.0, 4.0])
    z = se.range_bearing(truth, m)
    fix = se.position_fix(truth, np.zeros(2))
    xs, ys = np.linspace(-1, 6, 141), np.linspace(-2, 5, 141)
    sr, sb, sg = 0.15, 0.05, 0.5
    panels = {
        "range only": lambda p: se.measurement_log_likelihood(z[:1], se.range_bearing(p, m)[:1], np.eye(1) * sr**2),
        "bearing only": lambda p: se.measurement_log_likelihood(z[1:], se.range_bearing(p, m)[1:],
                                                                np.eye(1) * sb**2, [0]),
        "range and bearing": lambda p: se.measurement_log_likelihood(z, se.range_bearing(p, m),
                                                                     np.diag([sr**2, sb**2]), [1]),
        "position fix": lambda p: se.measurement_log_likelihood(fix, se.position_fix(p, np.zeros(2)),
                                                                np.eye(2) * sg**2),
    }
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, (title, loglik) in zip(axes, panels.items()):
        L = np.array([[loglik(np.array([x, y, truth[2]])) for x in xs] for y in ys])
        ax.imshow(np.exp(L - L.max()), origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="viridis")
        ax.plot(*m, "*", color=plotting.COLORS["landmark"], ms=14, mec="k")
        ax.plot(*truth[:2], "o", color="w", ms=5, mec="k")
        ax.set_title(title)
    fig.suptitle("Likelihood p(z | x, y, θ) of one reading over the robot position (heading known)")
    save(fig, "03_likelihoods.png")


# ---------------------------------------------------------------- chapter 4
@figure("04")
def grid_localization():
    landmarks = np.array([[2.0, 2.0], [2.0, 6.0], [10.0, 2.0], [10.0, 6.0], [6.0, 4.0], [8.5, 6.5]])
    dd, dt, k = se.DifferentialDrive(0.1, 0.5), 0.1, 1e-4
    u = np.vstack([np.tile([0.5, 0.0], (60, 1)), np.tile([0.5, 0.5], (31, 1)), np.tile([0.5, 0.0], (80, 1))])
    x0 = np.array([3.0, 1.0, 0.0])
    truth = se.integrate_controls(x0, u, dt)
    odom = se.run_dead_reckoning(x0, np.zeros((3, 3)), se.simulate_wheel_travel(u, dt, dd, k, se.Rng(1)), dd, k).poses
    sensor = se.RangeBearingSensor(max_range=4.0, fov=2 * np.pi, sigma_range=0.1, sigma_bearing=0.05)
    grid = se.GridLocalization(se.GridSpec(0, 12, 0, 8, 0.2, 36))
    rng = se.Rng(2)
    snapshots = {}
    for kk in range(len(u) + 1):
        if kk > 0:
            grid.predict_odometry(odom[kk - 1], odom[kk], [0.02, 0.002, 0.02, 0.002])
        if kk % 5 == 0:
            grid.update_range_bearing_anonymous(sensor.observe(truth[kk], landmarks, rng), landmarks, sensor.R())
        if kk in (0, 60, 120, len(u)):
            snapshots[kk] = grid.marginal_xy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (kk, m) in zip(axes, snapshots.items()):
        ax.imshow(m ** 0.5, origin="lower", extent=(0, 12, 0, 8), cmap="Blues")
        ax.plot(landmarks[:, 0], landmarks[:, 1], "*", color=plotting.COLORS["landmark"], ms=11, mec="k")
        ax.plot(truth[: kk + 1, 0], truth[: kk + 1, 1], "-", color=plotting.COLORS["truth"], lw=1)
        ax.plot(*truth[kk, :2], "o", color="w", mec="k", ms=5)
        ax.set_title(f"step {kk}")
    fig.suptitle("Global localisation with indistinguishable landmarks: √ of the (x, y) marginal of the grid belief")
    save(fig, "04_grid_localization.png")


def main(argv: list[str]) -> int:
    for chapter in sorted(FIGURES):
        if argv and chapter not in argv:
            continue
        for fn in FIGURES[chapter]:
            fn()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
