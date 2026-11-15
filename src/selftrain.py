"""Deterministic logistic regression for self-training on algorithmic pseudo-labels.

Used twice: a ball-free possession hint and a ball-free dead-ball hint, each trained on frames
where the ball-based evidence is confident and applied where it is not. No human label is
involved at any point (RULES.md: "pseudo-labels produced by an algorithm"). Newton iterations
from a zero start, so the result is a pure function of the data.
"""

import numpy as np


def fit_logistic(X, y, l2=1.0, iters=25):
    """Weights (bias last) for standardised X [n, f] and y in {0, 1}."""
    n, f = X.shape
    A = np.column_stack([X, np.ones(n)])
    w = np.zeros(f + 1)
    reg = np.full(f + 1, l2)
    reg[-1] = 0.0
    for _ in range(iters):
        z = np.clip(A @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = A.T @ (p - y) + reg * w
        hess = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def selftrain_llr(feats, train_mask, target, min_per_class, stride=1):
    """Log-likelihood ratio (class 1 vs class 0) per row of feats, or None if too few labels.

    feats [T, F]; train_mask [T] bool; target [T] in {0, 1} (read only where train_mask).
    The training class balance is subtracted, so the output is evidence, not a posterior.
    Rows with a NaN feature get 0.
    """
    valid = np.all(np.isfinite(feats), axis=1)
    rows = np.flatnonzero(train_mask & valid)[::max(1, stride)]
    y = target[rows].astype(np.float64)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if min(n0, n1) < min_per_class:
        return None
    mu = feats[rows].mean(axis=0)
    sd = feats[rows].std(axis=0)
    sd[sd < 1e-9] = 1.0
    w = fit_logistic((feats[rows] - mu) / sd, y)
    z = np.zeros(len(feats))
    z[valid] = np.column_stack([(feats[valid] - mu) / sd, np.ones(int(valid.sum()))]) @ w
    z[valid] -= np.log(n1 / float(n0))
    return z
