"""Run every agent over the sampled contracts (cached; stops at the budget).

Usage:
    python scripts/02_run_agents.py --dry-run          # first 5 contracts, all agents
    python scripts/02_run_agents.py                    # all contracts
    python scripts/02_run_agents.py --agents sonnet    # one agent only
Needs ANTHROPIC_API_KEY / OPENAI_API_KEY in .env. Not needed to reproduce the analysis:
the cached responses in outputs/raw/ are enough.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from crc.agents import Budget, keys_present, run_agent
from crc.data import load_sample

ROOT = Path(__file__).resolve().parent.parent


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--agents", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    agents = {k: v for k, v in cfg["agents"].items() if not args.agents or k in args.agents}
    missing = keys_present({v["provider"] for v in agents.values()})
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}")
    sample = load_sample()[: 5 if args.dry_run else None]
    budget = Budget(cfg["pricing"], cfg["budget"]["hard_stop_usd"])
    print(f"{len(sample)} contracts x {len(agents)} agents; already spent ${budget.spent:.2f}")

    # One worker pool per agent keeps each provider's rate limits separate and the cache warm.
    spent = defaultdict(float)
    jobs = [(a, c) for a in agents for c in sample]
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_agent, a, agents[a], c, budget): (a, c["id"]) for a, c in jobs}
        for f in tqdm(as_completed(futs), total=len(futs)):
            a, cid = futs[f]
            try:
                spent[a] += f.result().get("usd", 0)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {a} {cid}: {e}")
    for a, usd in spent.items():
        print(f"  {a:<13} ${usd:.3f}")
    print(f"total spent so far ${budget.spent:.2f} (hard stop ${cfg['budget']['hard_stop_usd']})")


if __name__ == "__main__":
    main()
