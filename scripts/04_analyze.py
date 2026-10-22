"""Test every pre-registered hypothesis and draw the article's figures.

Reads results/histories_t*.csv and results/clusters_t*.csv (from 03_match.py).
Writes results/*.csv, results/hypotheses.json and figures/fig2..fig6.

Usage:  python scripts/04_analyze.py
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from crc.analysis import (PAIR_ORDER, multi_table, pair_table, recall_estimates, spearman, stopping_curve)
from crc.estimators import loglinear

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
COL = {"same model, same prompt": "#b0452f", "same model, other prompt": "#d2843a", "same family": "#c9a227",
       "other family": "#3a6d8c", "different method": "#1f5a42"}


def load(thr):
    H = pd.read_csv(RES / f"histories_t{thr}.csv")
    H["contract"] = H["item"].str.split("|").str[0]
    C = pd.read_csv(RES / f"clusters_t{thr}.csv")
    return H, C


_BOOT = {}


def boot_init(H, C, main, share, seed):
    """Give each worker process the data once, not once per resample (keeps memory low)."""
    _BOOT.update(H=H, C=C, main=main, share=share, seed=seed)


def boot_once(b):
    H, C, main, share, seed = (_BOOT[k] for k in ("H", "C", "main", "share", "seed"))
    rng = np.random.default_rng(seed + 1000 + b)
    contracts = H["contract"].unique()
    pick = rng.choice(contracts, len(contracts), replace=True)
    counts = pd.Series(pick).value_counts()
    Hb = pd.concat([H[H.contract == c].assign(_rep=r) for c, n in counts.items() for r in range(n)])
    Cb = pd.concat([C[C.contract == c].assign(_rep=r) for c, n in counts.items() for r in range(n)])
    df = recall_estimates(Hb, Cb, main, share, seed)
    return {k: float((df[k] - df.true_recall).abs().mean()) for k in ("pooled", "best_pair", "all_agents")}


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    thr0, main = cfg["matching"]["overlap_threshold"], cfg["main_agents"]
    share, seed = cfg["audit"]["share"], cfg["seed"]
    H, C = load(thr0)
    N = len(H)
    out, hyp = {}, {}

    # ---- agents
    summary = pd.DataFrame({"agent": main + ["haiku_specA", "haiku_specB"]})
    summary["recall"] = [H[a].mean() for a in summary.agent]
    summary.to_csv(RES / "agent_recall.csv", index=False)

    # ---- pairs (labelled)
    P = pair_table(H, main, by_category=False)
    Pc = pair_table(H, main, by_category=True)
    P.to_csv(RES / "pairs_total.csv", index=False)
    Pc.to_csv(RES / "pairs_by_category.csv", index=False)
    by_type = Pc.groupby("type").agg(median_rel_err=("rel_err", "median"), coverage=("covered", "mean"),
                                     median_dep=("dep_ratio", "median"), n=("rel_err", "size")).reindex(PAIR_ORDER)
    by_type.to_csv(RES / "pairs_by_type.csv")
    out["by_type"] = by_type.round(3).to_dict(orient="index")

    cross = Pc[Pc.type == "other family"]
    hyp["H1"] = {"coverage_cross_family": float(cross.covered.mean()), "supported": bool(cross.covered.mean() >= 0.5)}
    med = by_type.median_rel_err
    order_ok = bool(all(med.iloc[i] <= med.iloc[i + 1] for i in range(3)))
    hyp["H2"] = {"median_rel_err_by_type": med.round(3).to_dict(), "ordered_as_predicted": order_ok}

    M = multi_table(H, main)
    M.to_csv(RES / "multi_source.csv", index=False)
    Mc = M[M.category != "all"]
    het = Mc[Mc.method.isin(["chao_mh", "jackknife2"])].groupby("method").rel_err.median()
    hyp["H3"] = {"cross_family_median_rel_err": float(med["other family"]),
                 "heterogeneity_median_rel_err": het.round(3).to_dict(),
                 "supported": bool(med["other family"] < 0 and (het.abs() < abs(med["other family"])).all())}

    S = pair_table(H, ["haiku_specA", "haiku_specB"], by_category=True)
    S.to_csv(RES / "specialists_by_category.csv", index=False)
    hyp["H4"] = {"specialist_median_rel_err": float(S.rel_err.median()), "supported": bool(S.rel_err.median() > 0)}

    # ---- label-free, by matching threshold (H5)
    lf = []
    for thr in cfg["matching"]["sensitivity"]:
        Ht, Ct = load(thr)
        r = recall_estimates(Ht, Ct, main, share, seed)
        lf.append({"threshold": thr, "clusters": len(Ct), "N_loglinear_audited": r.attrs["N_loglinear"],
                   "N_true": len(Ht)})
    LF = pd.DataFrame(lf)
    LF.to_csv(RES / "label_free_by_threshold.csv", index=False)
    vals = LF.N_loglinear_audited.to_numpy()
    hyp["H5"] = {"N_by_threshold": dict(zip(LF.threshold.astype(str), np.round(vals, 1))),
                 "monotonic": bool(np.all(np.diff(vals) >= 0) or np.all(np.diff(vals) <= 0)),
                 "two_source_spread": float(P.N_hat.max() - P.N_hat.min())}

    allm = M[M.category != "all"]
    two_src_abs = Pc.rel_err.abs().median()
    hyp["H6"] = {"median_abs_rel_err_two_source": float(two_src_abs),
                 "loglinear": float(allm[allm.method == "loglinear"].rel_err.abs().median()),
                 "jackknife2": float(allm[allm.method == "jackknife2"].rel_err.abs().median())}
    hyp["H6"]["supported"] = bool(hyp["H6"]["loglinear"] < two_src_abs and hyp["H6"]["jackknife2"] < two_src_abs)

    rho1 = spearman(P.dep_ratio, P.N_hat / N)
    rho2 = spearman(P.dep_ratio, P.ci_width_rel)
    hyp["H7"] = {"spearman_dep_vs_estimate": rho1, "spearman_dep_vs_ci_width": rho2,
                 "supported": bool(rho1[0] < 0 and rho2[0] < 0)}

    R = recall_estimates(H, C, main, share, seed)
    top = P.sort_values("dep_ratio", ascending=False).iloc[0]
    found_pair = (top.a, top.b) in R.attrs["interactions"] or (top.b, top.a) in R.attrs["interactions"]
    hyp["H8"] = {"most_dependent_pair": [top.a, top.b], "label_free_interactions": R.attrs["interactions"],
                 "supported": bool(found_pair)}

    kw = Pc[Pc.type == "different method"]
    llm = Pc[Pc.type != "different method"]
    hyp["H9"] = {"median_dep_keyword_pairs": float(kw.dep_ratio.median()), "median_dep_llm_pairs": float(llm.dep_ratio.median()),
                 "median_abs_err_keyword_pairs": float(kw.rel_err.abs().median()),
                 "median_abs_err_llm_pairs": float(llm.rel_err.abs().median())}
    hyp["H9"]["supported"] = bool(hyp["H9"]["median_dep_keyword_pairs"] < hyp["H9"]["median_dep_llm_pairs"]
                                  and hyp["H9"]["median_abs_err_keyword_pairs"] < hyp["H9"]["median_abs_err_llm_pairs"])

    # ---- primary outcome (H0), with a contract bootstrap
    R.to_csv(RES / "recall_estimates.csv", index=False)
    mae = {k: float((R[k] - R.true_recall).abs().mean()) for k in ("pooled", "best_pair", "all_agents")}
    B = cfg["bootstrap"]["resamples"]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1), initializer=boot_init,
                             initargs=(H, C, main, share, seed)) as ex:
        boots = list(ex.map(boot_once, range(B), chunksize=10))
    bdf = pd.DataFrame(boots)
    bdf.to_csv(RES / "bootstrap_recall_mae.csv", index=False)
    diff = bdf.all_agents - bdf.pooled
    hyp["H0"] = {"mae": mae, "mae_ci": {k: [float(bdf[k].quantile(.025)), float(bdf[k].quantile(.975))] for k in bdf},
                 "all_minus_pooled_ci": [float(diff.quantile(.025)), float(diff.quantile(.975))],
                 "best_pair": list(R.attrs["best_pair"]), "audited_clusters": R.attrs["audited"],
                 "N_true": N, "N_pool": R.attrs["N_pool"], "N_loglinear": R.attrs["N_loglinear"],
                 "supported": bool(mae["all_agents"] < mae["pooled"])}

    # ---- stopping curve, labelling artefacts
    order = ["haiku_run1", "haiku_run2", "haiku_alt", "sonnet", "gpt_luna", "gpt_sol", "keyword"]
    SC = stopping_curve(H, order)
    SC.to_csv(RES / "stopping_curve.csv", index=False)
    sample = json.loads((ROOT / "data" / "sample.json").read_text(encoding="utf-8"))
    from crc.data import gold_items
    lens = {g["item"]: g["end"] - g["start"] for g in gold_items(sample)}
    short = H.item.map(lens) < 40
    Pshort = pair_table(H[~short], main, by_category=True).groupby("type").rel_err.median().reindex(PAIR_ORDER)
    out["labelling_artefacts"] = {"short_labels_lt40": int(short.sum()),
                                  "missed_by_all": int((H[main].sum(axis=1) == 0).sum()),
                                  "missed_by_all_short": int(((H[main].sum(axis=1) == 0) & short).sum()),
                                  "median_rel_err_by_type_without_short": Pshort.round(3).to_dict()}
    out["N_true"], out["hypotheses"] = N, hyp
    (RES / "hypotheses.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=lambda x: round(float(x), 3)))
    figures(P, Pc, M, SC, R, N)


def figures(P, Pc, M, SC, R, N):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # fig2: every pair's estimate vs the truth, grouped by how related the two agents are
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for i, t in enumerate(PAIR_ORDER):
        d = P[P.type == t]
        xs = i + np.linspace(-0.25, 0.25, len(d)) if len(d) > 1 else [i]
        ax.errorbar(xs, d.N_hat, yerr=[d.N_hat - d.lo, d.hi - d.N_hat], fmt="o", color=COL[t], ms=5, capsize=2)
    ax.axhline(N, color="black", lw=1)
    ax.text(len(PAIR_ORDER) - 0.5, N, f" true: {N:,}", va="bottom", ha="right")
    ax.set_xticks(range(len(PAIR_ORDER)), [t.replace(", ", ",\n") for t in PAIR_ORDER])
    ax.set_ylabel("estimated number of clauses")
    ax.set_title("Two-agent estimates of how many clauses exist, by how related the agents are", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig2_pairs_vs_truth.png"); plt.close(fig)

    # fig3: the agreement paradox
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for t in PAIR_ORDER:
        d = P[P.type == t]
        ax.scatter(d.dep_ratio, d.N_hat / N, s=200 * d.ci_width_rel / P.ci_width_rel.max(), color=COL[t], label=t, alpha=.85)
    ax.axhline(1, color="black", lw=1)
    ax.set_xlabel("how much more two agents agree than chance (observed ÷ expected overlap)")
    ax.set_ylabel("estimate ÷ true number")
    ax.set_title("The more two agents agree, the further the estimate falls", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_agreement_paradox.png"); plt.close(fig)

    # fig4: estimators compared (per-category absolute error)
    rows = [("median two-agent pair", Pc.rel_err.abs().median()),
            ("best pair (other method)", Pc[Pc.type == "different method"].rel_err.abs().median())]
    Mc = M[M.category != "all"]
    for m, lab in [("chao_mh", "Chao (all agents)"), ("jackknife2", "jackknife (all agents)"),
                   ("loglinear", "log-linear (all agents)")]:
        rows.append((lab, Mc[Mc.method == m].rel_err.abs().median()))
    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
    ax.barh([r[0] for r in rows][::-1], [100 * r[1] for r in rows][::-1], color="#1f5a42")
    ax.set_xlabel("median error vs the true count, % (per clause category)")
    ax.set_title("Which estimate gets closest", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig4_estimators.png"); plt.close(fig)

    # fig5: stopping curve
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.plot(SC.agents, SC.found, "o-", color="#3a6d8c", label="found by at least one agent")
    ax.plot(SC.agents[1:], SC.estimated[1:], "s--", color="#d2843a", label="estimated total")
    ax.axhline(N, color="black", lw=1, label=f"true total ({N:,})")
    ax.set_xticks(SC.agents, [f"+{a}" for a in SC.added], rotation=30, ha="right")
    ax.set_ylabel("clauses")
    ax.set_title("Adding agents: what you see vs what exists", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig5_adding_agents.png"); plt.close(fig)

    # fig6: recall estimates per agent
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    y = np.arange(len(R))
    ax.scatter(R.true_recall, y, color="black", zorder=3, label="true recall")
    ax.scatter(R.pooled, y, marker="x", color="#b0452f", label="pooled (common practice)")
    ax.scatter(R.all_agents, y, marker="s", color="#1f5a42", label="capture–recapture + 10% audit")
    ax.set_yticks(y, R.agent)
    ax.set_xlabel("recall")
    ax.set_title("Estimating each agent's recall without full labels", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig6_recall_estimates.png"); plt.close(fig)


if __name__ == "__main__":
    main()
