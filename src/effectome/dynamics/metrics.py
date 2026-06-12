"""Distances between connectivity matrices (graph-space geometry).

The dynamic effectome is a trajectory in graph space, and how we *cluster* that
trajectory into recurring states (Stage 3a) depends entirely on the metric we put
on it. The Frobenius distance treats each matrix as a flat vector; this module
also provides geometries that respect the structure of connectivity matrices:

* **Log-Euclidean / affine-invariant** distances for *symmetric positive-definite*
  connectivity (covariance/precision-type), which avoid the determinant
  "swelling" bias of Euclidean averaging (Pennec et al. 2006; Arsigny et al. 2006).
* **Gromov-Wasserstein** distance for *directed, weighted* graphs compared as
  metric-measure spaces, i.e. without assuming node correspondence
  (Memoli 2011; Peyre et al. 2016).

These back the metric-aware clustering in ``graph_states.py`` (Fréchet
quantization): log-Euclidean clustering uses a closed-form barycenter, while the
affine-invariant and Gromov-Wasserstein options cluster from a precomputed
distance matrix with k-medoids (discrete Fréchet barycenters / medoids).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Flat (Frobenius / cosine) geometry
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# SPD (Riemannian) geometry: log-Euclidean and affine-invariant
# --------------------------------------------------------------------------- #


def spd_project(w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Symmetrize and floor eigenvalues so ``w`` becomes symmetric positive definite.

    Connectivity matrices are generally directed; SPD metrics require a symmetric
    PD operand, obtained here by ``(w + w.T)/2`` followed by eigenvalue flooring.
    """
    s = 0.5 * (w + w.T)
    vals, vecs = np.linalg.eigh(s)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T


def _sym_matrix_function(w: np.ndarray, fun: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a scalar ``fun`` to the eigenvalues of a symmetric matrix."""
    vals, vecs = np.linalg.eigh(0.5 * (w + w.T))
    return (vecs * fun(vals)) @ vecs.T


def logm_spd(w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Matrix logarithm of an SPD projection of ``w``."""
    return _sym_matrix_function(spd_project(w, eps), np.log)


def expm_sym(m: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix (inverse of :func:`logm_spd`)."""
    return _sym_matrix_function(m, np.exp)


def log_euclidean_features(matrices: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Map each (N, N) matrix to its vectorized log (so Euclidean ops = log-Euclidean).

    K-means on these features clusters under the log-Euclidean metric, and the
    Euclidean centroid back-mapped by :func:`expm_sym` is the log-Euclidean
    Fréchet mean.
    """
    logs = np.stack([logm_spd(w, eps) for w in matrices])
    return vectorize(logs)


def log_euclidean_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Log-Euclidean Riemannian distance ``||log a - log b||_F``."""
    return float(np.linalg.norm(logm_spd(a, eps) - logm_spd(b, eps), ord="fro"))


def affine_invariant_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Affine-invariant Riemannian distance ``||log(a^{-1/2} b a^{-1/2})||_F``."""
    sa = spd_project(a, eps)
    vals, vecs = np.linalg.eigh(sa)
    inv_sqrt = (vecs * (1.0 / np.sqrt(np.maximum(vals, eps)))) @ vecs.T
    m = inv_sqrt @ spd_project(b, eps) @ inv_sqrt
    ev = np.linalg.eigvalsh(0.5 * (m + m.T))
    return float(np.linalg.norm(np.log(np.maximum(ev, eps))))


# --------------------------------------------------------------------------- #
# Gromov-Wasserstein geometry (directed graphs as metric-measure spaces)
# --------------------------------------------------------------------------- #


def _sinkhorn(cost: np.ndarray, p: np.ndarray, q: np.ndarray, epsilon: float, n_iter: int) -> np.ndarray:
    """Entropic optimal-transport plan for ``cost`` with marginals ``p, q``.

    The cost is shifted by its global minimum before exponentiating; this leaves
    the optimal plan unchanged (a constant added to the whole cost factors out of
    the Sinkhorn updates) but prevents underflow when costs are large.
    """
    k = np.exp(-(cost - cost.min()) / epsilon) + 1e-300
    u = np.ones_like(p)
    v = np.ones_like(q)
    for _ in range(n_iter):
        u = p / (k @ v)
        v = q / (k.T @ u)
    return u[:, None] * k * v[None, :]


def gromov_wasserstein_distance(
    c1: np.ndarray,
    c2: np.ndarray,
    epsilon: float = 0.05,
    max_iter: int = 200,
    sinkhorn_iter: int = 50,
    tol: float = 1e-6,
) -> float:
    """Entropic Gromov-Wasserstein distance between two structure matrices.

    ``c1, c2`` are (relational) structure matrices, e.g. absolute connectivity.
    Uniform node marginals are assumed. Implements the projected-gradient /
    Sinkhorn scheme of Peyre, Cuturi & Solomon (2016) for the squared loss.
    """
    a1 = np.abs(np.asarray(c1, dtype=np.float64))
    a2 = np.abs(np.asarray(c2, dtype=np.float64))
    n1, n2 = a1.shape[0], a2.shape[0]
    p = np.full(n1, 1.0 / n1)
    q = np.full(n2, 1.0 / n2)

    # Constant part of the squared-loss tensor: f1(C1) p 1^T + 1 q^T f2(C2)^T, f(x)=x^2.
    const = (a1**2) @ np.outer(p, np.ones(n2)) + np.outer(np.ones(n1), q) @ (a2**2).T
    # Scale the entropic regularization to the cost magnitude so behavior (and the
    # exact symmetry of the result) is invariant to the overall scale of the inputs.
    eff_eps = epsilon * (float(np.mean(np.abs(const))) or 1.0)
    t = np.outer(p, q)
    obj = float(np.sum(const * t))
    prev = np.inf
    for _ in range(max_iter):
        grad = const - 2.0 * (a1 @ t @ a2.T)  # L(C1, C2) (x) T for squared loss
        t = _sinkhorn(grad, p, q, eff_eps, sinkhorn_iter)
        obj = float(np.sum((const - a1 @ t @ a2.T) * t))
        if abs(prev - obj) < tol:
            break
        prev = obj
    return float(np.sqrt(max(obj, 0.0)))


# --------------------------------------------------------------------------- #
# Generic pairwise distances + k-medoids (for non-Euclidean geometries)
# --------------------------------------------------------------------------- #

_PAIRWISE = {
    "frobenius": lambda a, b, **k: frobenius_distance(a, b),
    "cosine": lambda a, b, **k: cosine_distance(a, b),
    "log_euclidean": lambda a, b, **k: log_euclidean_distance(a, b, k.get("eps", 1e-6)),
    "affine_invariant": lambda a, b, **k: affine_invariant_distance(a, b, k.get("eps", 1e-6)),
    "gromov_wasserstein": lambda a, b, **k: gromov_wasserstein_distance(
        a, b, k.get("gw_epsilon", 0.05), k.get("gw_max_iter", 200)
    ),
}


def pairwise_distances(matrices: np.ndarray, metric: str, **kwargs) -> np.ndarray:
    """Full (K, K) symmetric distance matrix under ``metric``."""
    if metric not in _PAIRWISE:
        raise ValueError(f"unknown metric '{metric}'; choices: {sorted(_PAIRWISE)}")
    fn = _PAIRWISE[metric]
    k = len(matrices)
    d = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d[i, j] = d[j, i] = fn(matrices[i], matrices[j], **kwargs)
    return d


def kmedoids(
    distance: np.ndarray, k: int, seed: int = 42, max_iter: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster from a precomputed distance matrix; return (labels, medoid_indices).

    A medoid is the within-cluster point minimizing summed distance: the discrete
    Fréchet barycenter under whatever metric produced ``distance``.
    """
    rng = np.random.default_rng(seed)
    n = distance.shape[0]
    medoids = rng.choice(n, size=k, replace=False)
    labels = np.argmin(distance[:, medoids], axis=1)
    for _ in range(max_iter):
        new_medoids = medoids.copy()
        for c in range(k):
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue
            costs = distance[np.ix_(members, members)].sum(axis=1)
            new_medoids[c] = members[int(np.argmin(costs))]
        new_labels = np.argmin(distance[:, new_medoids], axis=1)
        if np.array_equal(new_labels, labels) and np.array_equal(new_medoids, medoids):
            break
        labels, medoids = new_labels, new_medoids
    return labels.astype(np.int64), medoids
