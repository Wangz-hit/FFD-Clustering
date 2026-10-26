"""Fuzzy association-variable grouping (paper Section 2.2.1)."""
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class FuzzyGroups:
    memberships: np.ndarray
    groups: list[np.ndarray]
    centers: np.ndarray
    partition_coefficient: float
    classification_entropy: float


def fuzzy_variable_groups(data: np.ndarray, n_groups: int, *, m: float = 2.0,
                          seed: int = 0, max_iter: int = 150, tol: float = 1e-6) -> FuzzyGroups:
    """FCM over *variables*, treating each variable's observations as its vector.

    K-means provides the initial centers (paper Eq. 3); Eqs. (4)--(5)
    update memberships and centers. A variable belongs to group j when
    membership(i,j) > 1 / C. Groups can overlap (soft grouping).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError('data must be a finite [samples, variables] matrix')
    n_vars = data.shape[1]
    if not 2 <= n_groups <= n_vars or m <= 1:
        raise ValueError('invalid group count or fuzzifier')
    vectors = data.T.copy()
    centers = KMeans(n_clusters=n_groups, random_state=seed, n_init=10).fit(vectors).cluster_centers_
    eps = 1e-12
    for _ in range(max_iter):
        squared = np.sum((vectors[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        zero = squared <= eps
        # This branch handles an exact match to a cluster center.
        scores = np.maximum(squared, eps) ** (-1.0 / (m - 1.0))
        memberships = scores / scores.sum(axis=1, keepdims=True)
        if zero.any():
            for i in np.where(zero.any(axis=1))[0]:
                memberships[i, :] = zero[i] / zero[i].sum()
        weights = memberships ** m
        new_centers = (weights.T @ vectors) / np.maximum(weights.sum(axis=0)[:, None], eps)
        if np.max(np.abs(new_centers - centers)) < tol:
            centers = new_centers
            break
        centers = new_centers
    # Memberships must correspond to the converged centers.
    squared = np.sum((vectors[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    zero = squared <= eps
    scores = np.maximum(squared, eps) ** (-1.0 / (m - 1.0))
    memberships = scores / scores.sum(axis=1, keepdims=True)
    for i in np.where(zero.any(axis=1))[0]:
        memberships[i, :] = zero[i] / zero[i].sum()
    groups = [np.flatnonzero(memberships[:, j] > 1.0 / n_groups) for j in range(n_groups)]
    # Guard against an empty group for a degenerate data matrix.
    groups = [g if len(g) else np.array([int(np.argmax(memberships[:, j]))])
              for j, g in enumerate(groups)]
    pc = float(np.mean(np.sum(memberships ** 2, axis=1)))
    ce = float(-np.mean(np.sum(memberships * np.log(np.maximum(memberships, eps)), axis=1)))
    return FuzzyGroups(memberships, groups, centers, pc, ce)


def group_count_diagnostics(data: np.ndarray, candidates: range, seed: int = 0) -> list[dict]:
    """Return PC/CE; the paper chooses C by inspecting these curves, not a fixed rule."""
    return [{'groups': c,
             'partition_coefficient': (r := fuzzy_variable_groups(data, c, seed=seed)).partition_coefficient,
             'classification_entropy': r.classification_entropy} for c in candidates]
