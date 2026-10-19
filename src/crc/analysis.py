"""Analysis for PREREGISTRATION.md: estimates, hypothesis checks and the recall-estimation outcome.

Two settings:
- labelled: rows are lawyer-labelled clauses (true N known), columns = agents (caught or not);
- label-free: rows are clusters of agents' quotes (false positives included), plus a
  simulated 10% audit that tells, for audited clusters only, whether they are real.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .estimators import chao_mh, chapman, jackknife_mh, loglinear

FAMILY = {"haiku": "anthropic", "sonnet": "anthropic", "gpt": "openai", "keyword": "rules"}


def family(agent: str) -> str:
    return next(v for k, v in FAMILY.items() if agent.startswith(k))


def pair_type(a: str, b: str) -> str:
    s = {a, b}
    if "keyword" in s:
        return "different method"
    if s == {"haiku_run1", "haiku_run2"}:
        return "same model, same prompt"
    if "haiku_alt" in s and s & {"haiku_run1", "haiku_run2"}:
        return "same model, other prompt"
    if family(a) == family(b):
        return "same family"
    return "other family"


PAIR_ORDER = ["same model, same prompt", "same model, other prompt", "same family", "other family", "different method"]


# ---------- labelled setting ----------

def pair_table(H: pd.DataFrame, agents: list[str], by_category: bool) -> pd.DataFrame:
    """Chapman per agent pair (optionally per category) vs the true number of labelled clauses."""
    groups = H.groupby("category") if by_category else [("all", H)]
    rows = []
    for cat, g in groups:
        N = len(g)
        for a, b in combinations(agents, 2):
            x, y = g[a].to_numpy(bool), g[b].to_numpy(bool)
            n1, n2, m = int(x.sum()), int(y.sum()), int((x & y).sum())
            e = chapman(n1, n2, m)
            exp_m = n1 * n2 / N
            rows.append({"category": cat, "a": a, "b": b, "type": pair_type(a, b), "N_true": N,
                         "n1": n1, "n2": n2, "m": m, "found": e.found, "N_hat": e.N, "lo": e.lo, "hi": e.hi,
                         "rel_err": e.N / N - 1, "covered": e.lo <= N <= e.hi,
                         "ci_width_rel": (e.hi - e.lo) / e.N if e.N > 0 else np.nan,
                         "dep_ratio": m / exp_m if exp_m else np.nan})
    return pd.DataFrame(rows)


def multi_source(H: np.ndarray, names: list[str], weights=None) -> dict:
    out = {}
    for label, fn in [("chao_mh", lambda: chao_mh(H, weights)),
                      ("jackknife1", lambda: jackknife_mh(H, 1, weights)),
                      ("jackknife2", lambda: jackknife_mh(H, 2, weights)),
                      ("loglinear", lambda: loglinear(H, names=names, weights=weights))]:
        e = fn()
        out[label] = e
    return out


def multi_table(H: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    rows = []
    for cat, g in [("all", H)] + list(H.groupby("category")):
        X = g[agents].to_numpy(int)
        seen = X[X.any(axis=1)]
        for k, e in multi_source(seen, agents).items():
            rows.append({"category": cat, "method": k, "N_true": len(g), "found": int(len(seen)),
                         "N_hat": e.N, "rel_err": e.N / len(g) - 1,
                         "interactions": e.detail.get("interactions")})
    return pd.DataFrame(rows)


def stopping_curve(H: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Found-by-anyone and estimated total as agents are added one at a time."""
    rows = []
    for k in range(1, len(order) + 1):
        X = H[order[:k]].to_numpy(int)
        seen = X[X.any(axis=1)]
        est = np.nan
        if k == 2:
            a, b = X[:, 0].astype(bool), X[:, 1].astype(bool)
            est = chapman(int(a.sum()), int(b.sum()), int((a & b).sum())).N
        elif k >= 3:
            est = jackknife_mh(seen, 2).N
        rows.append({"agents": k, "added": order[k - 1], "found": int(len(seen)), "estimated": est, "true": len(H)})
    return pd.DataFrame(rows)


# ---------- label-free setting ----------

def audit_weights(C: pd.DataFrame, agents: list[str], share: float, seed: int) -> tuple[np.ndarray, dict]:
    """Simulated human audit of a random share of clusters. Returns, for every cluster, the
    estimated probability that it is a real clause, by how many agents caught it (k)."""
    rng = np.random.default_rng(seed)
    k = C[agents].sum(axis=1).to_numpy()
    audited = rng.random(len(C)) < share
    real = C["is_real"].to_numpy(bool)
    overall = real[audited].mean() if audited.any() else 1.0
    p_by_k = {}
    for kk in np.unique(k):
        sel = audited & (k == kk)
        p_by_k[int(kk)] = real[sel].mean() if sel.sum() >= 5 else overall
    return np.array([p_by_k[int(x)] for x in k]), {"audited": int(audited.sum()), "p_by_k": p_by_k}


def recall_estimates(H: pd.DataFrame, C: pd.DataFrame, agents: list[str], share: float, seed: int) -> pd.DataFrame:
    """Primary outcome: each agent's recall estimated without labels, three ways, vs true recall."""
    N_true = len(H)
    true_r = {a: H[a].mean() for a in agents}
    w, audit = audit_weights(C, agents, share, seed)
    X = C[agents].to_numpy(int)
    n_w = {a: float((X[:, j] * w).sum()) for j, a in enumerate(agents)}
    pool = float(w.sum())

    # (c) all agents, log-linear on audit-weighted pattern counts
    ll = loglinear(X, names=agents, weights=w)
    # (b) the pair the label-free model calls least dependent: lowest observed/expected overlap
    #     using the log-linear N as the stand-in for the unknown truth
    best, best_ratio = None, np.inf
    for i, j in combinations(range(len(agents)), 2):
        m = float((X[:, i] * X[:, j] * w).sum())
        ratio = m / (n_w[agents[i]] * n_w[agents[j]] / ll.N)
        if ratio < best_ratio:
            best, best_ratio = (i, j), ratio
    i, j = best
    m = float((X[:, i] * X[:, j] * w).sum())
    pair_N = chapman(n_w[agents[i]], n_w[agents[j]], m).N

    rows = []
    for a in agents:
        rows.append({"agent": a, "true_recall": true_r[a],
                     "pooled": n_w[a] / pool,
                     "best_pair": n_w[a] / pair_N,
                     "all_agents": n_w[a] / ll.N})
    df = pd.DataFrame(rows)
    df.attrs.update({"N_true": N_true, "N_pool": pool, "N_pair": pair_N, "N_loglinear": ll.N,
                     "best_pair": (agents[i], agents[j]), "interactions": ll.detail["interactions"], **audit})
    return df


def spearman(x, y) -> tuple[float, float]:
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)
