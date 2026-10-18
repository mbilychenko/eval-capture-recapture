"""Capture-recapture estimators for "how many items exist, including the ones nobody found".

Each source (a map, an annotator, an AI agent) is one "catch". Items found by
several sources are "recaptures". All functions take plain counts or a 0/1
capture-history matrix (rows = items found by at least one source, columns =
sources), so they work for forests, fish or LLM outputs alike.

References
- Chapman (1951), the small-sample version of Lincoln-Petersen.
- Chao (1987), lower bound for heterogeneous catchability (model Mh).
- Burnham & Overton (1978), jackknife estimator for model Mh.
- Fienberg (1972), log-linear models for multiple-list (multi-source) data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np


@dataclass
class Estimate:
    N: float                     # estimated total number of items
    lo: float = float("nan")     # 95% interval
    hi: float = float("nan")
    found: int = 0               # items seen by at least one source
    method: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def missed(self) -> float:
        return self.N - self.found


def lincoln_petersen(n1: int, n2: int, m: int) -> Estimate:
    """Classic two-source estimate N = n1 * n2 / m (biased for small m)."""
    return Estimate(N=n1 * n2 / m if m else float("inf"), found=n1 + n2 - m, method="lincoln_petersen")


def chapman(n1: int, n2: int, m: int, z: float = 1.96) -> Estimate:
    """Two-source estimate with Chapman's small-sample correction and a normal 95% interval.

    n1, n2: items found by each source; m: items found by both.
    """
    N = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    se = var ** 0.5
    return Estimate(N=N, lo=N - z * se, hi=N + z * se, found=n1 + n2 - m, method="chapman")


def _freq(H: np.ndarray) -> tuple[int, int, np.ndarray]:
    """S = items seen, t = sources, f[j] = items seen by exactly j sources (j = 1..t)."""
    H = np.asarray(H, dtype=int)
    k = H.sum(axis=1)
    k = k[k > 0]
    t = H.shape[1]
    f = np.bincount(k, minlength=t + 1)
    return len(k), t, f


def chao_mh(H: np.ndarray) -> Estimate:
    """Chao (1987) lower bound for model Mh: items differ in how easy they are to catch.

    Uses the bias-corrected form N = S + (t-1)/t * f1 (f1 - 1) / (2 (f2 + 1)),
    which stays finite when no item is seen exactly twice (classic form: S + f1^2 / (2 f2)).
    """
    S, t, f = _freq(H)
    f1, f2 = int(f[1]), int(f[2]) if t >= 2 else 0
    N = S + (t - 1) / t * f1 * (f1 - 1) / (2 * (f2 + 1))
    return Estimate(N=N, found=S, method="chao_mh", detail={"f1": int(f1), "f2": int(f2), "t": t})


def jackknife_mh(H: np.ndarray, order: int = 1) -> Estimate:
    """Burnham & Overton (1978) jackknife for model Mh, first or second order."""
    S, t, f = _freq(H)
    if order == 1:
        N = S + (t - 1) / t * f[1]
    elif order == 2:
        if t < 3:
            raise ValueError("second-order jackknife needs at least 3 sources")
        N = S + (2 * t - 3) / t * f[1] - (t - 2) ** 2 / (t * (t - 1)) * f[2]
    else:
        raise ValueError("order must be 1 or 2")
    return Estimate(N=N, found=S, method=f"jackknife{order}_mh", detail={"t": t})


def loglinear(H: np.ndarray, max_interactions: int | None = None, names: list[str] | None = None) -> Estimate:
    """Multi-source log-linear estimate (Fienberg 1972) with AIC-selected pairwise dependence.

    Counts each observed capture pattern, fits a Poisson GLM with main effects plus
    the pairwise interactions that lower AIC (added greedily), and predicts the
    count of the unobservable all-zero pattern: exp(intercept).
    Needs at least 3 sources; with 2 sources dependence can't be identified.
    """
    import statsmodels.api as sm

    H = np.asarray(H, dtype=int)
    t = H.shape[1]
    if t < 3:
        raise ValueError("log-linear models need at least 3 sources")
    names = names or [f"s{i}" for i in range(t)]
    patterns = [p for p in product([0, 1], repeat=t) if any(p)]
    counts = {p: 0 for p in patterns}
    for row in H:
        if row.any():
            counts[tuple(row)] += 1
    X_main = np.array(patterns, dtype=float)
    y = np.array([counts[p] for p in patterns], dtype=float)
    pairs = list(combinations(range(t), 2))

    def fit(chosen):
        cols = [X_main] + [(X_main[:, i] * X_main[:, j])[:, None] for i, j in chosen]
        X = sm.add_constant(np.hstack(cols), has_constant="add")
        return sm.GLM(y, X, family=sm.families.Poisson()).fit()

    chosen: list[tuple[int, int]] = []
    best = fit(chosen)
    limit = len(pairs) if max_interactions is None else max_interactions
    while len(chosen) < limit:
        trials = [(fit(chosen + [p]), p) for p in pairs if p not in chosen]
        if not trials:
            break
        res, p = min(trials, key=lambda r: r[0].aic)
        if res.aic >= best.aic - 2:          # require a real improvement
            break
        best, chosen = res, chosen + [p]
    b0, se0 = best.params[0], best.bse[0]
    S = int(y.sum())
    return Estimate(N=S + np.exp(b0), lo=S + np.exp(b0 - 1.96 * se0), hi=S + np.exp(b0 + 1.96 * se0),
                    found=S, method="loglinear",
                    detail={"interactions": [(names[i], names[j]) for i, j in chosen], "aic": float(best.aic)})


def pair_counts(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
    """n1, n2, m from two 0/1 vectors over the same items."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return int(a.sum()), int(b.sum()), int((a & b).sum())


def dependence_ratio(a: np.ndarray, b: np.ndarray, N_true: int) -> float:
    """Observed overlap / overlap expected if the two sources were independent.

    > 1 means the sources find the SAME items more often than chance:
    they share blind spots, and two-source estimates come out too low.
    """
    n1, n2, m = pair_counts(a, b)
    expected = n1 * n2 / N_true
    return m / expected if expected else float("nan")
