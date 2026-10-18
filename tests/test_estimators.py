"""Checks for the estimators: known values, plus simulations where the truth is known."""
import numpy as np
import pytest

from crc import chao_mh, chapman, dependence_ratio, jackknife_mh, lincoln_petersen, loglinear


def test_chapman_forest_numbers():
    e = chapman(1052, 1190, 727)
    assert e.found == 1515
    assert round(e.N) == 1722
    assert (round(e.lo), round(e.hi)) == (1678, 1765)


def test_lincoln_petersen_textbook():
    # 100 marked, 100 caught, 20 marked among them -> 500 fish
    assert lincoln_petersen(100, 100, 20).N == 500


def simulate(N, p, shared_ease=0.0, t=2, seed=0):
    """Capture histories for N items. shared_ease > 0 gives items a common 'easiness'
    that raises their catch chance in EVERY source: the sources share blind spots."""
    rng = np.random.default_rng(seed)
    ease = rng.normal(0, shared_ease, N) if shared_ease else np.zeros(N)
    logit = np.log(p / (1 - p)) + ease
    prob = 1 / (1 + np.exp(-logit))
    H = (rng.random((N, t)) < prob[:, None]).astype(int)
    return H


def test_independent_sources_are_roughly_unbiased():
    ests = []
    for s in range(300):
        H = simulate(1000, 0.4, seed=s)
        ests.append(chapman(*_pair(H)).N)
    assert abs(np.mean(ests) / 1000 - 1) < 0.02


def test_shared_blind_spots_underestimate():
    ests, ratios = [], []
    for s in range(300):
        H = simulate(1000, 0.4, shared_ease=2.0, seed=s)
        ests.append(chapman(*_pair(H)).N)
        ratios.append(dependence_ratio(H[:, 0], H[:, 1], 1000))
    assert np.mean(ests) < 900           # at least 10% too low
    assert np.mean(ratios) > 1.1         # and the sources overlap more than chance


def test_mh_estimators_reduce_the_bias():
    H = simulate(1000, 0.3, shared_ease=1.5, t=6, seed=1)
    seen = H[H.any(axis=1)]
    two = chapman(*_pair(H)).N
    assert abs(chao_mh(seen).N - 1000) < abs(two - 1000)
    assert abs(jackknife_mh(seen, 2).N - 1000) < abs(two - 1000)


def test_loglinear_recovers_n_with_one_dependent_pair():
    rng = np.random.default_rng(3)
    N = 2000
    a = rng.random(N) < 0.4
    b = np.where(a, rng.random(N) < 0.7, rng.random(N) < 0.2)   # b depends on a
    c = rng.random(N) < 0.4                                       # c independent
    H = np.c_[a, b, c].astype(int)
    seen = H[H.any(axis=1)]
    e = loglinear(seen, names=["a", "b", "c"])
    assert abs(e.N / N - 1) < 0.08
    assert ("a", "b") in e.detail["interactions"]
    naive = chapman(*_pair(H[:, :2]))
    assert naive.N < 0.9 * N                                      # the dependent pair alone is far off


def _pair(H):
    a, b = H[:, 0].astype(bool), H[:, 1].astype(bool)
    return int(a.sum()), int(b.sum()), int((a & b).sum())


def test_loglinear_needs_three_sources():
    with pytest.raises(ValueError):
        loglinear(np.ones((5, 2), int))
