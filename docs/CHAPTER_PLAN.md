# Chapter plan — learn-state-estimation   (`<pkg>` = `state_estimation`)

Probabilistic state estimation and SLAM for mobile robots. Existing material to mine: differential-drive dead
reckoning, KF/EKF localization, particle filter, feature-based EKF SLAM. The course-provided Python framework
(`PR_LAB4-main` and similar) is NOT kept; write the filters from scratch in C++.

1. Probability for robotics: Gaussians, covariance propagation, Bayes' rule, Mahalanobis distance.
2. Motion models: differential drive, velocity vs odometry model, dead reckoning and its drift.
3. Measurement models: range-bearing, Cartesian features, GNSS-like position fixes.
4. The Bayes filter and histogram (grid) localization.
5. The Kalman filter: derivation, consistency checks (NEES/NIS).
6. The EKF: map-based localization with known features; linearization errors (break it).
7. The particle filter: sampling, weighting, resampling strategies, particle deprivation.
8. Data association: nearest neighbour, individual compatibility, JCBB.
9. Feature-based EKF SLAM: state augmentation, covariance growth, loop closure.
10. Pose-based SLAM with scan matching: ICP (point-to-point, point-to-line), pose graphs, intro to
    factor graphs (compare against GTSAM in a notebook).

Used in: [pekf-slam-icp](https://github.com/Rudip1/pekf-slam-icp).
Acknowledgement: chapters 6 and 9 grew out of joint work with Gebrecherkos Gebreslassie.
