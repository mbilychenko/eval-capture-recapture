"""A non-AI checker: keyword rules per clause type.

It splits the contract into sentences and quotes every sentence that matches one of
the rules for a clause type. It has very different blind spots from an LLM: it never
paraphrases, catches headings and boilerplate, and misses clauses worded unusually.
It costs nothing and needs no API key. The rules were written once, before the full
run, from general contract vocabulary (not tuned on the labels).
"""
from __future__ import annotations

import re

RULES: dict[str, list[str]] = {
    "License Grant": [
        r"\bhereby grants?\b.{0,200}\blicen[cs]e",
        r"\bgrants?\b.{0,80}\b(a|an)\b.{0,80}\blicen[cs]e\b",
        r"\blicen[cs]e[sd]? (to|under)\b.{0,60}\b(use|make|sell|reproduce|distribute)",
    ],
    "Cap On Liability": [
        r"\bin no event\b.{0,200}\bliab",
        r"\bliability\b.{0,150}\b(shall not exceed|not to exceed|limited to|in excess of|exceed the)\b",
        r"\baggregate liability\b",
        r"\blimitation of liability\b",
    ],
    "Anti-Assignment": [
        r"\b(may not|shall not|will not|cannot|not be)\b.{0,80}\bassign",
        r"\bassign(ed|ment|able)?\b.{0,120}\bwithout\b.{0,60}\bconsent\b",
        r"\bnon-?assignable\b",
    ],
    "Audit Rights": [
        r"\baudit",
        r"\b(inspect|examine)\b.{0,80}\b(books|records|accounts)\b",
    ],
    "Insurance": [
        r"\binsurance\b",
        r"\binsured\b",
    ],
    "Post-Termination Services": [
        r"\b(upon|after|following|on) (the )?(any )?(termination|expiration)\b",
        r"\bsurviv(e|es|al)\b.{0,60}\b(termination|expiration)\b",
    ],
    "Minimum Commitment": [
        r"\bminimum\b.{0,60}\b(purchase|order|quantit|amount|commitment|volume|payment|fee)",
        r"\bat least\b.{0,40}\b(units|orders|purchases)\b",
    ],
    "Revenue/Profit Sharing": [
        r"\broyalt(y|ies)\b",
        r"\b(revenue|profit)s?[- ]shar",
        r"\b(percent|%)\b.{0,40}\b(of )?(net |gross )?(sales|revenues?|profits|receipts)\b",
    ],
    "Exclusivity": [
        r"\bexclusive(ly)?\b",
        r"\bsole (and exclusive )?(supplier|distributor|provider|source)\b",
    ],
    "Ip Ownership Assignment": [
        r"\bhereby (irrevocably )?assigns?\b",
        r"\b(sole and exclusive |exclusive |sole )?propert(y|ies) of\b",
        r"\bown(s)? all right, title and interest\b",
        r"\bwork(s)? made for hire\b",
    ],
}

_COMPILED = {k: [re.compile(p, re.I | re.S) for p in v] for k, v in RULES.items()}
_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\"“])|\n\s*\n")


def sentences(text: str) -> list[str]:
    out = []
    for s in _SPLIT.split(text):
        s = " ".join(s.split())
        if 25 <= len(s) <= 2000:
            out.append(s)
    return out


def keyword_answer(text: str, categories: list[str]) -> dict[str, list[str]]:
    sents = sentences(text)
    return {cat: [s for s in sents if any(p.search(s) for p in _COMPILED[cat])] for cat in categories}
