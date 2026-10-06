"""Exact balanced-binary empirical erasers in column-vector convention.

An endpoint row x is edited as x @ matrix.T + bias. MP uses the fitted
mean difference; SAL uses an independently formed centered cross-covariance
SVD; LEACE uses the pseudoinverse of the empirical pooled covariance.
No ridge or truncation beyond numerical rank detection is used.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FittedEraser:
    name: str
    matrix: np.ndarray
    bias: np.ndarray
    covariance_support_error: float = 0.0

    def transform(self, rows):
        return np.asarray(rows) @ self.matrix.T + self.bias

    @property
    def linear_cost(self):
        return float(np.square(self.matrix - np.eye(self.matrix.shape[0])).sum())

    @property
    def affine_cost(self):
        # Reference V ~ N(0, I), not the empirical pooled distribution.
        return self.linear_cost + float(self.bias @ self.bias)


def fit_erasers(h_plus, h_minus):
    """Fit MP, centered rank-one SAL, and exact binary LEACE to N pairs.

    MP and SAL return linear maps. LEACE includes its empirical-mean
    preserving affine offset. All have the same deleted direction, but their
    distortion relative to a fixed zero-mean isotropic reference may differ.
    """
    h_plus = np.asarray(h_plus, dtype=np.float64)
    h_minus = np.asarray(h_minus, dtype=np.float64)
    if h_plus.ndim != 2 or h_minus.shape != h_plus.shape:
        raise ValueError("Expected equally shaped N-by-d endpoint arrays.")
    n, d = h_plus.shape
    if n < 1 or d < 2 or not np.isfinite(h_plus).all() or not np.isfinite(h_minus).all():
        raise ValueError("Finite endpoints with N >= 1 and d >= 2 are required.")
    eye = np.eye(d)
    mean_difference = h_plus.mean(axis=0) - h_minus.mean(axis=0)
    if np.linalg.norm(mean_difference) == 0:
        raise ValueError("This experiment excludes the zero empirical contrast.")
    w_mp = mean_difference / np.linalg.norm(mean_difference)
    mp = FittedEraser("MP", eye - np.outer(w_mp, w_mp), np.zeros(d))

    endpoints = np.concatenate((h_plus, h_minus), axis=0)
    mean = endpoints.mean(axis=0)
    centered = endpoints - mean
    labels = np.concatenate((np.ones(n), -np.ones(n)))
    labels -= labels.mean()
    cross_covariance = (centered.T @ labels / len(labels))[:, None]
    left, _, _ = np.linalg.svd(cross_covariance, full_matrices=False)
    w_sal = left[:, 0]
    sal = FittedEraser("SAL", eye - np.outer(w_sal, w_sal), np.zeros(d))

    # SVD of centered endpoints avoids squaring the condition number during
    # numerical rank detection. The eigenvalues of Sigma are singular² / 2N.
    _, singular, right_t = np.linalg.svd(centered, full_matrices=False)
    cutoff = max(centered.shape) * np.finfo(np.float64).eps * singular[0]
    keep = singular > cutoff
    right = right_t[keep].T
    eigenvalues = singular[keep] ** 2 / len(endpoints)
    v = cross_covariance[:, 0]
    support_error = float(np.linalg.norm(v - right @ (right.T @ v)) / np.linalg.norm(v))
    if support_error > 1e-10:
        raise ArithmeticError("Numerical covariance rank truncation removed concept support.")
    sigma_plus_v = right @ ((right.T @ v) / eigenvalues)
    denominator = float(v @ sigma_plus_v)
    if denominator <= 0:
        raise ArithmeticError("The empirical concept covariance must be nonzero.")
    a = sigma_plus_v / denominator
    matrix = eye - np.outer(v, a)
    leace = FittedEraser("LEACE", matrix, (eye - matrix) @ mean, support_error)
    return {fit.name: fit for fit in (mp, sal, leace)}


def stable_rowspace_residual(matrix, direction):
    """Compute ||Pi_row(M) u||² directly from the SVD of M.

    Rank is detected on M, not on M M.T. This avoids promoting roundoff in
    the erased singular direction to a nonzero covariance eigenvalue.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    _, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    cutoff = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    keep = singular > cutoff
    return float(np.square(right_t[keep] @ direction).sum())
