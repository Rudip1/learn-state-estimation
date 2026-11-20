# 8 · Data association

Chapters 4–7 assumed that every reading arrived labelled with the landmark that produced it. Real sensors do not
label their readings: a laser sees a corner, a camera a blob. **Data association** decides which map feature, if
any, produced each reading. It is the most fragile part of feature-based estimation: one wrong pairing fed into
an EKF pulls the estimate towards a wrong pose *and* shrinks the covariance, so the next associations are made
with false confidence and errors cascade. This chapter builds up from the naive nearest neighbour to joint
compatibility branch and bound (JCBB).

Code: [`data_association.hpp`](../cpp/include/state_estimation/data_association.hpp) ·
Tests: [`test_data_association.cpp`](../cpp/tests/test_data_association.cpp) ·
Notebook: [`08_data_association.ipynb`](../2_notebooks/solutions/08_data_association.ipynb)

## 8.1 The problem

A scan gives $n$ readings $z_1, \dots, z_n$; the map has $N$ features. A **hypothesis** assigns to each reading a
feature or nothing:

$$
\mathcal H = \{j_1, \dots, j_n\},\qquad j_i \in \{1, \dots, N\} \cup \{\ast\}, \tag{8.1}
$$

where $\ast$ marks a spurious reading (clutter) or a feature not yet in the map (chapter 9). Each feature produces
at most one reading per scan. There are more than $(N+1)^n$ hypotheses; we need a criterion to rank them and a way to
search them.

All methods below use the linearised predictions of the filter: for feature $j$, the predicted reading
$\hat z_j = h_j(\bar x)$ and its Jacobian $H_j$ with respect to the state, whose covariance is $\bar P$. In localisation
the state is the pose; in SLAM it also contains the features, and $H_j$ has non-zero blocks for both.

## 8.2 Nearest neighbour and individual compatibility

The **nearest neighbour** (NN) pairs each reading with the feature whose prediction is closest, $\min_j \lVert z_i -
\hat z_j\rVert$, within a fixed distance. It ignores uncertainty: a reading 0.5 m from a prediction is "close" whether the
robot is sure of its position to 1 cm or to 1 m. (In range–bearing space the norm even mixes metres and radians.)

**Individual compatibility** (IC) measures distance in units of the uncertainty. The innovation of pairing reading $i$
with feature $j$ and its covariance are

$$
\nu_{ij} = z_i - \hat z_j \quad(\text{bearing wrapped}), \tag{8.2}
$$

$$
S_{ij} = H_j\bar PH_j^T + R. \tag{8.3}
$$

If $z_i$ really comes from feature $j$, then $\nu_{ij} \sim \mathcal N(0, S_{ij})$ and its squared Mahalanobis distance is
$\chi^2_m$ (§1.6). The pairing is **individually compatible** if

$$
D^2_{ij} = \nu_{ij}^TS_{ij}^{-1}\nu_{ij} < \chi^2_m(1-\alpha), \tag{8.4}
$$

and **ICNN** pairs each reading with the individually compatible feature of smallest $D^2_{ij}$ (the report this chapter
grew from used exactly this rule inside map-based EKF localisation). ICNN is cheap, $O(nN)$, and works well when the
robot is well localised. It has two weaknesses: it does not stop two readings from claiming the same feature, and,
more fundamentally, it judges every pairing in isolation.

## 8.3 Why individual compatibility is not enough

The innovations of different readings are **correlated**: they all depend on the same uncertain robot pose. If the
robot is in fact 0.9 m further along a corridor than it believes, *every* reading is shifted by 0.9 m in the same
direction. Each reading, judged alone, may be closest to the wrong feature — the one 1 m behind its true source —
and the wrong pairings all pass the individual gate because the pose uncertainty is large along the corridor.

**Worked example (tested).** Four landmarks 1 m apart on a line; the robot position estimate has standard deviation
1 m along the line and 0.1 m across; the robot is actually 0.9 m ahead of its estimate; readings have 0.1 m noise.
ICNN pairs readings 2, 3 and 4 with landmarks 1, 2 and 3 (and reading 1, which has no closer candidate, with
landmark 1 too). Every one of those pairings is individually compatible, but together they require the robot to be
shifted by $-0.1$ m and by $+0.9$ m at the same time — the joint test rejects them. Drop reading 1, and the shifted
pairings of readings 2–4 *are* jointly compatible (one $-0.1$ m shift explains them all). JCBB prefers the correct
hypothesis because it explains all four readings: the maximum-pairings criterion matters as much as the joint test. The correct hypothesis, all four readings with their own landmarks and a single 0.9 m shift
of the robot, is jointly compatible.

## 8.4 Joint compatibility

Stack the innovations of all pairings in a hypothesis $\mathcal H$ with $k$ paired readings:

$$
\nu_\mathcal H = \begin{bmatrix}\nu_{1j_1}\\ \vdots\\ \nu_{kj_k}\end{bmatrix},\qquad
H_\mathcal H = \begin{bmatrix}H_{j_1}\\ \vdots\\ H_{j_k}\end{bmatrix},\qquad
S_\mathcal H = H_\mathcal H\bar PH_\mathcal H^T + \operatorname{diag}(R, \dots, R). \tag{8.5}
$$

The off-diagonal blocks $H_{j_a}\bar PH_{j_b}^T$ of $S_\mathcal H$ are the correlations ICNN throws away. The joint
NIS

$$
D^2_\mathcal H = \nu_\mathcal H^TS_\mathcal H^{-1}\nu_\mathcal H \tag{8.6}
$$

is $\chi^2_{mk}$ if all pairings are right, and the hypothesis is **jointly compatible** if

$$
D^2_\mathcal H < \chi^2_{mk}(1-\alpha). \tag{8.7}
$$

**JCBB** (Neira and Tardós, 2001) searches the tree of hypotheses depth first, reading by reading. At level $i$ it tries
every unused feature $j$ that is individually compatible with $z_i$ *and* keeps the partial hypothesis jointly
compatible, then the branch where $z_i$ is left unpaired. Among complete hypotheses it keeps the one with the most
pairings (ties: the smallest $D^2_\mathcal H$). Two facts make the search fast:

- Joint compatibility is monotone in practice: if a partial hypothesis fails (8.7), adding pairings rarely rescues it,
  so its subtree is cut.
- Bound: a branch whose pairings so far plus the readings still to place cannot beat the best hypothesis found is cut.

The worst case is exponential, but with a well-localised robot most branches die at the first level. The tests
compare JCBB with exhaustive search on random problems with clutter.

**Choosing α.** A small $\alpha$ (wide gates) accepts more correct pairings in noisy conditions but also more wrong
ones; $\alpha = 0.05$ is common for localisation, smaller values when clutter is rare.

## 8.5 Using the association

Once $\mathcal H$ is chosen:

- Paired readings go into one stacked EKF update (6.11) — exactly the stack whose joint NIS was just tested.
- Unpaired readings are clutter in localisation, or candidate new features in SLAM (chapter 9).
- A scan with too few pairings (or a joint NIS near the gate) is a warning sign: the robot may be lost, and
  global methods (chapters 4 and 7) are needed.

## Algorithm summary — JCBB

1. Predict $\hat z_j$, $H_j$ for every feature that could be visible.
2. Compute $D^2_{ij}$ (8.4) for all pairs; keep the individually compatible ones.
3. Depth-first search over readings: for reading $i$, try each unused compatible feature, keep the branch if the
   partial hypothesis passes (8.7); then try leaving $i$ unpaired, if the bound allows.
4. Return the hypothesis with the most pairings, ties by smallest joint NIS.
5. Update the filter with all paired readings at once.

## Common mistakes

- Gating with a fixed Euclidean distance (NN), which ignores how uncertain the robot is.
- Letting two readings pair with the same feature.
- Testing pairings independently when the robot pose is uncertain (ICNN) — correlated innovations make individual
  tests over-optimistic.
- Forgetting to wrap the bearing in the innovation, which makes a perfect pairing look incompatible.
- Updating the filter with a pairing that passed the gate only because the gate was set very wide.

## References

- J. Neira, J. D. Tardós, "Data association in stochastic mapping using the joint compatibility test",
  *IEEE Trans. Robotics and Automation* 17(6), 2001.
- Y. Bar-Shalom, T. E. Fortmann, *Tracking and Data Association*, Academic Press, 1988 — gating, nearest neighbour.
- S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005 — §7.5 (unknown correspondences), §10.3.
- J. A. Castellanos, J. D. Tardós, *Mobile Robot Localization and Map Building*, Kluwer, 1999.
