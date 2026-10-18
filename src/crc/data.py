"""CUAD loading and sampling.

CUAD_v1.json is SQuAD-style: one entry per contract, one question per clause
category, and lawyer-labelled answer spans (character offsets into the contract text).
A gold "item" here is one labelled clause: (contract, category, start, end).
Overlapping labels of the same category in the same contract are merged into one item.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "CUAD_v1.json"
SAMPLE = ROOT / "data" / "sample.json"


def load_cuad(path: Path = RAW) -> list[dict]:
    """[{id, text, categories: {name: {"definition": str, "spans": [(start, end), ...]}}}]"""
    raw = json.loads(path.read_text(encoding="utf-8"))["data"]
    out = []
    for c in raw:
        p = c["paragraphs"][0]
        cats = {}
        for qa in p["qas"]:
            m = re.search(r'related to "(.+?)".*?Details: (.*)$', qa["question"], flags=re.S)
            name, definition = m.group(1), m.group(2).strip()
            spans = sorted({(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]})
            cats[name] = {"definition": definition, "spans": merge_spans(spans)}
        out.append({"id": c["title"], "text": p["context"], "categories": cats})
    return out


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def draw_sample(contracts: list[dict], categories: list[str], n: int, max_chars: int, seed: int) -> list[dict]:
    """Seeded random sample of contracts no longer than max_chars, keeping only the chosen categories."""
    pool = [c for c in contracts if len(c["text"]) <= max_chars]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False))
    return [{"id": pool[i]["id"], "text": pool[i]["text"],
             "categories": {k: pool[i]["categories"][k] for k in categories}} for i in pick]


def load_sample(path: Path = SAMPLE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def gold_items(sample: list[dict]) -> list[dict]:
    """Flat list of gold items: {item, contract, category, start, end}."""
    items = []
    for c in sample:
        for cat, v in c["categories"].items():
            for s, e in v["spans"]:
                items.append({"item": f"{c['id']}|{cat}|{s}", "contract": c["id"], "category": cat,
                              "start": s, "end": e})
    return items
