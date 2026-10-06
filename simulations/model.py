"""Standalone Gaussian repair calculations. Only NumPy and SciPy are needed."""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import beta, chi2, poisson


def poisson_terms(d, kappa):
    if kappa == 0:
        return np.array([0.]), np.array([1.])
    mu = kappa / 2
    lo, hi = poisson.ppf([1e-14, 1-1e-14], mu).astype(int)
    j = np.arange(max(0, lo), hi+1, dtype=float)
    return j, np.exp(-mu+j*np.log(mu)-gammaln(j+1))


def exact_risk(d, kappa, k=1):
    if k == 0:
        return 1.
    j, weights = poisson_terms(d, kappa)
    return (d-k)*math.fsum((weights/(d+2*j)).tolist())


def integral_risk(d, kappa, k=1):
    scale = d+kappa
    value, error = quad(lambda s: np.exp(-d*s/scale+
                       .5*kappa*np.expm1(-2*s/scale)), 0, np.inf,
                       epsabs=2e-12, epsrel=2e-12, limit=200)
    assert error < 1e-8
    return (d-k)*value/scale


def exact_operating(d, kappa, threshold, epsilon):
    """Joint bad action and weighted loss via independent radius/Beta mixture."""
    j, weights = poisson_terms(d, kappa)
    radius_tail = chi2.sf(threshold, d+2*j)
    a, b = (d-1)/2, (1+2*j)/2
    action = float(weights @ radius_tail)
    joint_bad = float((weights*radius_tail) @ beta.sf(epsilon, a, b))
    weighted_loss = float((weights*radius_tail) @ (a/(a+b)))
    return dict(action=action, joint_bad=joint_bad,
                loss_including_abstention=1-action+weighted_loss,
                loss_on_action=weighted_loss/action if action else None)


def probability(values):
    count, n = int(np.count_nonzero(values)), len(values)
    lower = 0. if count == 0 else float(beta.ppf(.025, count, n-count+1))
    upper = 1. if count == n else float(beta.ppf(.975, count+1, n-count))
    return dict(count=count, n=n, estimate=count/n, ci95=[lower, upper])


def mean_summary(values):
    return dict(mean=float(np.mean(values)),
                mc_se=float(np.std(values, ddof=1)/np.sqrt(len(values))), n=len(values))


def policy_summary(act, residual=None):
    result = dict(action=probability(act))
    if residual is not None:
        result.update(joint_bad=probability(act & (residual > .5)),
                      loss_including_abstention=mean_summary(np.where(act, residual, 1.)),
                      loss_on_action=mean_summary(residual[act]) if act.sum()>1 else None)
    return result


def ellipsoid_dual(x, omega, alpha=.05, epsilon=.5):
    """Feasible upper dual; omega contains diagonal calibration variances."""
    norm2 = np.sum(x*x, axis=1)
    v = x/np.sqrt(np.maximum(norm2, np.finfo(float).tiny))[:, None]
    lv = v*np.sqrt(omega)
    a = (1-epsilon)*np.diag(omega)[None,:,:]-lv[:,:,None]*lv[:,None,:]
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    b = -epsilon*x*np.sqrt(omega)
    coordinates = np.einsum('nij,ni->nj', eigenvectors, b)
    squared = coordinates**2
    c = chi2.ppf(1-alpha, x.shape[1])
    gap = 1e-12*max(1., float(omega.max()))
    lo = np.maximum(0., eigenvalues[:,-1])+gap
    hi = lo+np.linalg.norm(b, axis=1)/np.sqrt(c)+gap
    for _ in range(60):
        mid = (lo+hi)/2
        outside = np.sum(squared/(mid[:,None]-eigenvalues)**2, axis=1)>c
        lo = np.where(outside, mid, lo)
        hi = np.where(outside, hi, mid)
    correction = np.sum(squared/(hi[:,None]-eigenvalues), axis=1)
    upper = -epsilon*norm2+hi*c+correction
    margin = 1e-10*(1+epsilon*norm2+hi*c+correction)
    return dict(dual=hi, upper=upper, margin=margin,
                eigenmax=eigenvalues[:,-1], act=(upper+margin<0)&(norm2>0))
