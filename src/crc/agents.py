"""Contract-review agents: one LLM pass that quotes every clause of the requested types.

Each agent = (provider, model, prompt style, temperature). Responses are cached to
outputs/raw/<agent>/<contract>.json while running, then packed into one committed file,
outputs/agent_responses.jsonl.gz, so re-running never pays twice and the
analysis can be reproduced from it with no API keys. Every call's tokens
and cost go to outputs/ledger.csv; the run stops before it crosses the budget.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_OUT = ROOT / "outputs" / "raw"                     # per-call cache while running (not committed)
BUNDLE = ROOT / "outputs" / "agent_responses.jsonl.gz"  # all responses in one committed file
LEDGER = ROOT / "outputs" / "ledger.csv"

SYSTEM_STANDARD = (
    "You are a contract review agent working for a lawyer. You will receive a contract and a list of "
    "clause types. For each clause type, find EVERY passage in the contract that belongs to it. "
    "Quote each passage verbatim, exactly as written in the contract (copy the text, do not paraphrase). "
    "Keep each quote to the sentence or sentences that make up that clause. "
    "If a clause type does not appear, return an empty list for it.\n\n"
    "Answer with JSON only, in this shape: {\"<clause type>\": [\"<verbatim quote>\", ...], ...}"
)

SYSTEM_CHECKLIST = (
    "Act as a careful second-pass reviewer. Work through the contract section by section, from the first "
    "heading to the signature block, and keep a checklist of the clause types below. Every time a section "
    "contains one of them, copy the exact wording into that clause type's list. Exact wording only, no "
    "summaries. One entry per distinct clause; leave a list empty when the contract has none.\n\n"
    "Output: a single JSON object mapping each clause type name to a list of exact quotes. No other text."
)

PROMPTS = {"standard": SYSTEM_STANDARD, "checklist": SYSTEM_CHECKLIST}


def focus_text(focus: list[str]) -> str:
    """Specialist instruction: search some clause types hard, report the others only when obvious."""
    return ("\n\nYou are a specialist in these clause types: " + "; ".join(focus) + ". Search for them with "
            "extra care, including clauses hidden in unexpected sections. For the other clause types, report "
            "only clauses that are obvious at first reading.")


def task_text(categories: dict[str, dict]) -> str:
    lines = [f"- {name}: {v['definition']}" for name, v in categories.items()]
    return "Clause types to find:\n" + "\n".join(lines)


def parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        return json.loads(m.group(0)) if m else {}


class Budget:
    def __init__(self, prices: dict, hard_stop: float):
        self.prices, self.hard_stop = prices, hard_stop
        self.spent = 0.0
        self.lock = threading.Lock()
        if LEDGER.exists():
            with LEDGER.open(encoding="utf-8") as f:
                self.spent = sum(float(r["usd"]) for r in csv.DictReader(f))

    def cost(self, model: str, usage: dict) -> float:
        p = self.prices[model]
        return (usage["input"] * p["input"] + usage.get("cache_read", 0) * p.get("cache_read", p["input"])
                + usage.get("cache_write", 0) * p.get("cache_write", p["input"]) + usage["output"] * p["output"]) / 1e6

    def record(self, agent: str, contract: str, model: str, usage: dict) -> float:
        usd = self.cost(model, usage)
        with self.lock:
            return self._write(agent, contract, model, usage, usd)

    def _write(self, agent, contract, model, usage, usd):
        self.spent += usd
        new = not LEDGER.exists()
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["agent", "contract", "model", "input", "cache_read", "cache_write", "output", "usd"])
            w.writerow([agent, contract, model, usage["input"], usage.get("cache_read", 0),
                        usage.get("cache_write", 0), usage["output"], f"{usd:.6f}"])
        return usd

    def check(self):
        if self.spent >= self.hard_stop:
            raise RuntimeError(f"budget hard stop: ${self.spent:.2f} spent (limit ${self.hard_stop})")


def call_anthropic(model, system, contract_text, task, temperature, max_tokens=8000, **_):
    import anthropic
    client = anthropic.Anthropic()
    # Contract first, cached: the same contract is read by several Anthropic agents.
    r = client.messages.create(
        model=model, max_tokens=max_tokens,   # SDK 1.x: no temperature; models use their default sampling
        system=[{"type": "text", "text": "CONTRACT:\n" + contract_text, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": system}],
        messages=[{"role": "user", "content": task}],
    )
    u = r.usage
    usage = {"input": u.input_tokens, "output": u.output_tokens,
             "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
             "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}
    return "".join(b.text for b in r.content if b.type == "text"), usage


def call_openai(model, system, contract_text, task, temperature, max_tokens=16000, reasoning_effort=None):
    from openai import OpenAI
    client = OpenAI()
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_completion_tokens=max_tokens, **extra,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": "CONTRACT:\n" + contract_text + "\n\n" + system},
                  {"role": "user", "content": task}],
    )
    u = r.usage
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    usage = {"input": u.prompt_tokens - cached, "cache_read": cached, "output": u.completion_tokens}
    return r.choices[0].message.content or "", usage


CALLERS = {"anthropic": call_anthropic, "openai": call_openai}


_bundle: dict | None = None
_bundle_lock = threading.Lock()


def bundled() -> dict:
    """All committed responses from outputs/agent_responses.jsonl.gz, keyed by (agent, contract id)."""
    global _bundle
    with _bundle_lock:
        if _bundle is None:
            _bundle = {}
            if BUNDLE.exists():
                with gzip.open(BUNDLE, "rt", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        _bundle[(rec["agent"], rec["contract"])] = rec
    return _bundle


def load_cached(agent: str, contract_id: str) -> dict | None:
    """A response from this machine's outputs/raw/ cache, else from the committed bundle."""
    f = RAW_OUT / agent / f"{safe(contract_id)}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return bundled().get((agent, contract_id))


def pack_cache() -> int:
    """Write every response (bundle + outputs/raw/) into the single committed bundle file."""
    recs = dict(bundled())
    for f in RAW_OUT.glob("*/*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        recs[(rec["agent"], rec["contract"])] = rec
    with gzip.open(BUNDLE, "wt", encoding="utf-8") as out:
        for key in sorted(recs):
            out.write(json.dumps(recs[key], ensure_ascii=False) + "\n")
    return len(recs)


def run_agent(name: str, spec: dict, contract: dict, budget: Budget, retries: int = 3) -> dict:
    """Return the cached or fresh response for one agent on one contract."""
    cached = load_cached(name, contract["id"])
    if cached is not None:
        return cached
    out = RAW_OUT / name / f"{safe(contract['id'])}.json"
    if spec["provider"] == "keyword":                      # rule-based checker: no API, no cost
        from .keyword_agent import keyword_answer
        rec = {"agent": name, "contract": contract["id"], "model": "keyword-rules", "prompt": None,
               "focus": None, "temperature": None, "usage": {"input": 0, "output": 0}, "usd": 0.0,
               "answer": keyword_answer(contract["text"], list(contract["categories"])), "raw_text": ""}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec
    budget.check()
    system = PROMPTS[spec["prompt"]] + (focus_text(spec["focus"]) if spec.get("focus") else "")
    task = task_text(contract["categories"])
    last = None
    for attempt in range(retries):
        try:
            opts = {k: spec[k] for k in ("reasoning_effort",) if spec.get(k)}
            text, usage = CALLERS[spec["provider"]](spec["model"], system, contract["text"], task,
                                                    spec["temperature"], **opts)
            parsed = parse_json(text)
            usd = budget.record(name, contract["id"], spec["model"], usage)
            rec = {"agent": name, "contract": contract["id"], "model": spec["model"], "prompt": spec["prompt"],
                   "focus": spec.get("focus"), "temperature": spec["temperature"], "usage": usage, "usd": round(usd, 6),
                   "answer": {k: list(v) if isinstance(v, list) else [] for k, v in parsed.items()},
                   "raw_text": text}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            return rec
        except Exception as e:  # noqa: BLE001 - network/rate-limit/parse errors: retry
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{name} failed on {contract['id']}: {last}")


def safe(s: str) -> str:
    """Short, unique file name for a contract id (CUAD titles exceed Windows' path limit)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)[:50] + "_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def keys_present(providers: set[str]) -> list[str]:
    need = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
    return [need[p] for p in providers if p in need and not os.environ.get(need[p])]
