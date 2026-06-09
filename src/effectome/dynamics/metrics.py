"""Distances between connectivity matrices (graph-space geometry)."""

from __future__ import annotations

import numpy as np


def vectorize(matrices: np.ndarray) -> np.ndarray:
    """Flatten a (K, N, N) stack into (K, N*N) feature vectors."""
    k = matrices.shape[0]
    return matrices.reshape(k, -1)


def frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius norm of the difference between two matrices."""
    return float(np.linalg.norm(a - b, ord="fro"))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity between two flattened matrices."""
    av, bv = a.ravel(), b.ravel()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom == 0:
        return 1.0
    return float(1.0 - (av @ bv) / denom)


def pairwise_frobenius(matrices: np.ndarray) -> np.ndarray:
    """Full (K, K) Frobenius distance matrix over a stack of connectivity matrices."""
    feats = vectorize(matrices)
    sq = np.sum(feats**2, axis=1)
    gram = feats @ feats.T
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * gram, 0.0)
    return np.sqrt(d2)
