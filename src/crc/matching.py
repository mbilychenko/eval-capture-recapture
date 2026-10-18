"""Turn agent quotes into capture histories.

1. Locate each quote in the contract: exact match after whitespace normalisation,
   else the best fuzzy alignment (rapidfuzz) if it scores >= fuzzy_min_score.
2. With labels: a gold item is "caught" by an agent if one of its located quotes
   overlaps the gold span by >= threshold of the shorter of the two.
3. Without labels (what a practitioner sees): cluster all agents' quotes per
   contract and category by the same overlap rule; each cluster is one item.
"""
from __future__ import annotations

import re

import numpy as np
from rapidfuzz import fuzz

WS = re.compile(r"\s+")


def _norm_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace; return the normalised text and a map back to original offsets."""
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space and out:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
    return "".join(out), idx


class Locator:
    def __init__(self, text: str, fuzzy_min_score: float = 90):
        self.text = text
        self.norm, self.idx = _norm_map(text)
        self.fuzzy_min = fuzzy_min_score

    def find(self, quote: str) -> tuple[int, int, str] | None:
        q = WS.sub(" ", quote.strip()).lower().strip(' ."“”')
        if len(q) < 8:
            return None
        k = self.norm.find(q)
        if k >= 0:
            return self.idx[k], self.idx[k + len(q) - 1] + 1, "exact"
        al = fuzz.partial_ratio_alignment(q, self.norm, score_cutoff=self.fuzzy_min)
        if al is None:
            return None
        s, e = al.dest_start, max(al.dest_end - 1, al.dest_start)
        return self.idx[s], self.idx[e] + 1, "fuzzy"


def overlap_share(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    return inter / min(a[1] - a[0], b[1] - b[0])


def locate_all(sample, responses, agents, fuzzy_min_score=90):
    """-> list of predictions {agent, contract, category, start, end, how} and a located/unlocated tally."""
    preds, tally = [], {"exact": 0, "fuzzy": 0, "not_found": 0}
    for c in sample:
        loc = Locator(c["text"], fuzzy_min_score)
        for a in agents:
            rec = responses.get((a, c["id"]))
            if rec is None:
                continue
            for cat in c["categories"]:
                for q in rec["answer"].get(cat, []):
                    if not isinstance(q, str):
                        continue
                    hit = loc.find(q)
                    if hit is None:
                        tally["not_found"] += 1
                        continue
                    tally[hit[2]] += 1
                    preds.append({"agent": a, "contract": c["id"], "category": cat,
                                  "start": hit[0], "end": hit[1], "how": hit[2]})
    return preds, tally


def gold_histories(gold, preds, agents, threshold=0.5):
    """0/1 matrix (gold items x agents): did the agent catch the item? Plus per-agent precision counts."""
    by = {}
    for p in preds:
        by.setdefault((p["contract"], p["category"]), []).append(p)
    H = np.zeros((len(gold), len(agents)), dtype=int)
    col = {a: j for j, a in enumerate(agents)}
    matched_pred = set()
    for i, g in enumerate(gold):
        for k, p in enumerate(by.get((g["contract"], g["category"]), [])):
            if overlap_share((g["start"], g["end"]), (p["start"], p["end"])) >= threshold:
                H[i, col[p["agent"]]] = 1
                matched_pred.add(id(p))
    tp = {a: 0 for a in agents}
    n = {a: 0 for a in agents}
    for p in preds:
        n[p["agent"]] += 1
        tp[p["agent"]] += id(p) in matched_pred
    precision = {a: (tp[a] / n[a] if n[a] else float("nan")) for a in agents}
    return H, precision, n


def cluster_histories(preds, agents, threshold=0.5):
    """Label-free: union overlapping predictions (any agent) into items. Returns histories + cluster spans."""
    col = {a: j for j, a in enumerate(agents)}
    groups = {}
    for p in preds:
        groups.setdefault((p["contract"], p["category"]), []).append(p)
    rows, spans = [], []
    for key, ps in groups.items():
        parent = list(range(len(ps)))

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                if overlap_share((ps[i]["start"], ps[i]["end"]), (ps[j]["start"], ps[j]["end"])) >= threshold:
                    parent[root(i)] = root(j)
        clusters = {}
        for i, p in enumerate(ps):
            clusters.setdefault(root(i), []).append(p)
        for members in clusters.values():
            h = np.zeros(len(agents), dtype=int)
            for p in members:
                h[col[p["agent"]]] = 1
            rows.append(h)
            spans.append({"contract": key[0], "category": key[1],
                          "start": min(p["start"] for p in members), "end": max(p["end"] for p in members)})
    return (np.array(rows) if rows else np.zeros((0, len(agents)), int)), spans
