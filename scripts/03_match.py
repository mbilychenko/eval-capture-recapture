"""Match cached agent quotes to the contracts and to the lawyer labels.

Writes, for each matching threshold in config (matching.sensitivity):
  results/histories_t<thr>.csv   gold items x agents (0/1): who caught which labelled clause
  results/clusters_t<thr>.csv    label-free items x agents: what you'd see without labels
  results/agent_summary.csv      per agent: quotes, located share, precision, recall (primary threshold)

Usage:  python scripts/03_match.py [--dry-run]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from crc.agents import RAW_OUT, safe
from crc.data import gold_items, load_sample
from crc.matching import cluster_histories, gold_histories, locate_all, overlap_share

ROOT = Path(__file__).resolve().parent.parent


def load_responses(sample, agents):
    resp = {}
    for a in agents:
        for c in sample:
            f = RAW_OUT / a / f"{safe(c['id'])}.json"
            if f.exists():
                resp[(a, c["id"])] = json.loads(f.read_text(encoding="utf-8"))
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    m = cfg["matching"]
    agents = list(cfg["agents"])
    sample = load_sample()[: 5 if args.dry_run else None]
    resp = load_responses(sample, agents)
    done = {c["id"] for c in sample if all((a, c["id"]) in resp for a in agents)}
    if len(done) < len(sample):
        print(f"note: {len(sample) - len(done)} contracts lack a response from every agent and are excluded")
    sample = [c for c in sample if c["id"] in done]
    gold = gold_items(sample)
    preds, tally = locate_all(sample, resp, agents, m["fuzzy_min_score"])
    print(f"{len(sample)} contracts, {len(gold)} gold items, quotes located: {tally}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    suffix = "_dry" if args.dry_run else ""
    for thr in m["sensitivity"]:
        H, prec, n = gold_histories(gold, preds, agents, thr)
        g = pd.DataFrame(H, columns=agents)
        g.insert(0, "item", [x["item"] for x in gold])
        g.insert(1, "category", [x["category"] for x in gold])
        g.to_csv(out / f"histories_t{thr}{suffix}.csv", index=False)
        main = cfg["main_agents"]
        C, spans = cluster_histories([p for p in preds if p["agent"] in main], main, thr)
        cdf = pd.concat([pd.DataFrame(spans), pd.DataFrame(C, columns=main)], axis=1)
        # is_real: the cluster overlaps a labelled clause (used only for the simulated audit)
        gold_by = {}
        for x in gold:
            gold_by.setdefault((x["contract"], x["category"]), []).append((x["start"], x["end"]))
        cdf["is_real"] = [any(overlap_share((r.start, r.end), gsp) >= thr for gsp in gold_by.get((r.contract, r.category), []))
                          for r in cdf.itertuples()]
        cdf.to_csv(out / f"clusters_t{thr}{suffix}.csv", index=False)
        if thr == m["overlap_threshold"]:
            rows = [{"agent": a, "model": cfg["agents"][a]["model"], "prompt": cfg["agents"][a]["prompt"],
                     "quotes_located": n[a], "precision": round(prec[a], 3),
                     "recall": round(g[a].mean(), 3), "gold_caught": int(g[a].sum())} for a in agents]
            summary = pd.DataFrame(rows)
            summary.to_csv(out / f"agent_summary{suffix}.csv", index=False)
            print(summary.to_string(index=False))
            print(f"caught by at least one agent: {int(H.any(axis=1).sum())} of {len(gold)}")


if __name__ == "__main__":
    main()
