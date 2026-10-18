"""Matching on a real sampled contract: gold quotes must be located and caught."""
import pytest

from crc.data import SAMPLE, gold_items, load_sample
from crc.matching import Locator, cluster_histories, gold_histories, locate_all, overlap_share


def test_overlap_share():
    assert overlap_share((0, 10), (5, 30)) == 0.5
    assert overlap_share((0, 10), (10, 20)) == 0.0


@pytest.mark.skipif(not SAMPLE.exists(), reason="run scripts/01_sample.py first")
def test_gold_quotes_round_trip():
    c = next(c for c in load_sample() if any(v["spans"] for v in c["categories"].values()))
    gold = gold_items([c])
    # a fake agent that quotes every gold clause with messy whitespace, one paraphrase, one made-up quote
    answer = {}
    for g in gold:
        answer.setdefault(g["category"], []).append("  " + " \n ".join(c["text"][g["start"]:g["end"]].split()) + " ")
    first_cat = gold[0]["category"]
    answer[first_cat].append("This sentence does not appear anywhere in the contract at all.")
    resp = {("fake", c["id"]): {"answer": answer}, ("empty", c["id"]): {"answer": {}}}
    preds, tally = locate_all([c], resp, ["fake", "empty"])
    assert tally["not_found"] == 1
    H, prec, n = gold_histories(gold, preds, ["fake", "empty"], 0.5)
    assert H[:, 0].all() and not H[:, 1].any()
    C, spans = cluster_histories(preds, ["fake", "empty"], 0.5)
    assert len(spans) <= len(gold)


def test_fuzzy_locates_small_typos():
    text = "The Licensee shall maintain insurance coverage of not less than one million dollars per occurrence."
    hit = Locator(text).find("The Licensee shall maintain insurance coverage of not less than one milion dollars")
    assert hit is not None and hit[2] == "fuzzy"
