"""Draw the seeded contract sample and report the gold statistics.

Writes data/sample.json (contract text + lawyer-labelled spans for the chosen
categories; CUAD is CC BY 4.0, see DATA_NOTICE.md) and results/gold_stats.csv.

Usage:  python scripts/01_sample.py
"""
import json
from pathlib import Path

import pandas as pd
import yaml

from crc.data import SAMPLE, draw_sample, gold_items, load_cuad

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    s = cfg["sample"]
    sample = draw_sample(load_cuad(), s["categories"], s["n_contracts"], s["max_chars"], cfg["seed"])
    SAMPLE.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

    g = pd.DataFrame(gold_items(sample))
    per_c = g.groupby(["contract", "category"]).size()
    stats = g.groupby("category").agg(gold_items=("item", "size"), contracts=("contract", "nunique"))
    stats["max_per_contract"] = per_c.groupby("category").max()
    stats = stats.sort_values("gold_items", ascending=False)
    (ROOT / "results").mkdir(exist_ok=True)
    stats.to_csv(ROOT / "results" / "gold_stats.csv")
    chars = sum(len(c["text"]) for c in sample)
    print(stats.to_string())
    print(f"\n{len(sample)} contracts, {len(g)} gold items (true N), "
          f"{chars / 1e6:.2f}M characters (~{chars / 4 / 1e6:.2f}M tokens per agent run)")


if __name__ == "__main__":
    main()
