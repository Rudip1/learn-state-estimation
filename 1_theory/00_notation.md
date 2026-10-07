# Notation

One notation is used through the whole module. Vectors are bold-free lower case and are column vectors;
matrices are upper case. Each symbol is defined here once; chapters only repeat a definition where it helps.

## General

| Symbol | Meaning |
|---|---|
| $x \sim \mathcal N(\mu, \Sigma)$ | $x$ is Gaussian with mean $\mu$ and covariance $\Sigma$ |
| $\mathcal N(x;\mu,\Sigma)$ | the Gaussian density evaluated at $x$ |
| $E[\cdot]$, $\operatorname{Cov}[\cdot]$ | expectation, covariance |
| $p(a \mid b)$ | conditional probability (density) of $a$ given $b$ |
| $\eta$ | normalising constant of Bayes' rule |
| $d^2_M$ | squared Mahalanobis distance |
| $\chi^2_k(p)$ | $p$-quantile of the chi-square distribution with $k$ degrees of freedom |
| $I_n$, $0_{m\times n}$ | identity, zero matrix |
| $\operatorname{wrap}(\alpha)$ | angle wrapped to $(-\pi, \pi]$ |
| $J_f$ or $\partial f/\partial x$ | Jacobian of $f$ |

## Robot, map and sensors

| Symbol | Meaning |
|---|---|
| $k$ | discrete time step; $\Delta t$ the sampling period |
| $x_k = [x\;\, y\;\, \theta]^T$ | planar robot pose in the world frame $W$ |
| $\oplus$, $\ominus$ | pose compounding and inversion (Smith–Self–Cheeseman) |
| $J_{1\oplus}$, $J_{2\oplus}$, $J_\ominus$ | Jacobians of compounding w.r.t. first / second argument, and of inversion |
| $\boxplus$ | compounding of a pose with a point: $x \boxplus p$ maps a point from robot to world frame |
| $u_k$ | control input or odometry reading at step $k$ |
| $v$, $\omega$ | linear and angular velocity |
| $r$, $b$ | wheel radius, wheel base (distance between the wheels) |
| $m_j = [m_{j,x}\;\, m_{j,y}]^T$ | position of landmark $j$ in the world frame |
| $z_k$ | measurement at step $k$; $z_k^i$ the $i$-th observation in a scan |
| $\rho$, $\phi$ | range and bearing of a range–bearing observation |
| $c_k^i$ | correspondence: index of the landmark that produced $z_k^i$ |

## Filters

| Symbol | Meaning |
|---|---|
| $f(x_{k-1}, u_k, w_k)$ | motion (process) model |
| $h(x_k, v_k)$ | measurement (observation) model |
| $w_k \sim \mathcal N(0, Q_k)$ | process noise |
| $v_k \sim \mathcal N(0, R_k)$ | measurement noise |
| $\bar x_k, \bar P_k$ | predicted (prior) mean and covariance at $k$ |
| $\hat x_k, P_k$ | corrected (posterior) mean and covariance at $k$ |
| $F_k$, $W_k$ | Jacobians of $f$ w.r.t. state and process noise |
| $H_k$, $V_k$ | Jacobians of $h$ w.r.t. state and measurement noise |
| $\nu_k = z_k - h(\bar x_k)$ | innovation |
| $S_k$ | innovation covariance |
| $K_k$ | Kalman gain |
| $\epsilon_k$ (NEES), $\epsilon_{\nu,k}$ (NIS) | normalised estimation / innovation error squared |
| $x^{[i]}_k$, $w^{[i]}_k$, $M$ | particle $i$, its importance weight, number of particles |
| $N_\text{eff}$ | effective sample size |

## SLAM and scan matching

| Symbol | Meaning |
|---|---|
| $y_k = [x_k^T\;\, m_1^T \dots m_n^T]^T$ | EKF-SLAM state: robot pose followed by $n$ landmarks |
| $g(x_k, z)$ | inverse observation model (landmark from pose and measurement) |
| $G_x$, $G_z$ | Jacobians of $g$ |
| $R$, $t$ | rotation and translation of a rigid transform (scan matching) |
| $x_{0:T}$ | sequence of poses of a pose graph |
| $e_{ij}$, $\Omega_{ij}$ | error and information matrix of the edge between poses $i$ and $j$ |
