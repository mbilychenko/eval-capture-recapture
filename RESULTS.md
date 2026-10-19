# Results (full run, 2026-10-19)

All numbers come from `results/` (regenerate with `scripts/03_match.py` then `scripts/04_analyze.py`; no API keys needed). The hypotheses are in `PREREGISTRATION.md`; each one is reported here, including those that failed.

## Run
- 294 contracts × 9 agents; every call succeeded. **$14.71 in total** (`outputs/ledger.csv`).
- 19,283 quotes: 99.6% located in the contracts (18,824 exact, 374 fuzzy, 85 not found).
- True N = 1,413 lawyer-labelled clauses. **127 (9.0%) were caught by none of the 7 main agents.**

| Agent | Recall | Precision* |
|---|---|---|
| Claude Haiku 5.5, run 1 / run 2 / checklist prompt | 0.749 / 0.742 / 0.747 | 0.51 / 0.51 / 0.49 |
| Claude Sonnet 5.5 | 0.749 | 0.50 |
| GPT-6 luna | 0.745 | 0.44 |
| GPT-6.1 sol | **0.860** | 0.41 |
| Keyword rules (no AI) | 0.534 | 0.33 |

\*Precision against the lawyer labels. Part of the "false positives" are real clauses the labels don't contain (see Limits).

## Every pair underestimates, and agreement sets how much (H2: supported)
Chapman estimates for all 21 pairs fall below the true 1,413. The more related the two agents, the further below:

| Pair type | Total estimate (pairs) | Median error per category | Dependence ratio |
|---|---|---|---|
| Same model, same prompt | 1,102 | −19.0% | 1.24 |
| Same model, other prompt | 1,110–1,113 | −18.7% | 1.23 |
| Same family | 1,149–1,243 | −12.9% | 1.15 |
| Other family | 1,173–1,258 | −10.6% | 1.12 |
| Different method (keyword + LLM) | 1,193–1,264 | −10.4% | 1.12 |

- No pair is independent: the dependence ratio is 1.12–1.28 for all 21 pairs.
- Within OpenAI, luna and sol are barely more alike than cross-family pairs (1.14). The Claude models are clearly alike (1.22–1.28).

## The intervals are narrow and wrong (H1: not supported)
- The cross-family intervals contain the true per-category count in **1.2%** of the 80 checks.
- Keyword pairs manage 26.7%.
- The two-source interval only measures sampling noise. It knows nothing about shared blind spots, so it reports ±5 clauses around an estimate that is 300 too low.

## Most of the gap is "famous forests", and estimators built for it close it (H3: supported)
- Even agents from different families and methods underestimate (about −10%), because some clauses are obvious to everyone and others are hard for everyone.
- The estimators that allow for this cut the median per-category error to **−4.1%** (second-order jackknife) and **−4.3%** (Chao).

## More agents beat any pair (H6: supported)
- Median absolute error per category: two-agent pairs 12.1%; all 7 agents with the log-linear model 5.3%, or the jackknife 5.3%.
- Totals with all 7 agents:
  - log-linear 1,360 (−3.8%);
  - jackknife2 1,349 (−4.5%);
  - jackknife1 1,342 (−5.0%);
  - Chao 1,313 (−7.1%).

## Primary outcome: estimating each agent's recall without labels (H0: supported)
- **Pooled recall** (everything any agent found = the truth) **overstates every agent's recall by about 10 points**. Haiku shows 0.86 against a true 0.75.
- **Capture–recapture with all 7 agents plus a 10% simulated audit** (408 clusters checked):

  | | All agents + audit | Pooled | Best pair (gpt_sol + keyword) |
  |---|---|---|---|
  | Mean absolute error, this run | **1.8 points** | 10.2 points | 9.6 points |
  | Median over 1,000 contract resamples | **3.8 points** (90% range 1.3–9.5) | 9.7 | — |

  The all-agent method beats pooling in **94%** of resamples. Its pre-registered 95% interval for the difference (−9.1 to +1.9 points) just includes zero.
- The audit-corrected label-free total is 1,166 (−17.5%). It undercounts clauses, but the recall *ratios* stay accurate because numerator and denominator come from the same clusters.

## The rest
- **H4 (specialists overestimate): not supported.** Two Haiku agents told to focus on different halves still underestimate (−18.3%). Telling the same model to specialise doesn't make it a different net.
- **H5 (matching moves the label-free estimate): supported.** 1,187 / 1,166 / 1,096 at thresholds 0.3 / 0.5 / 0.7, a spread smaller than the spread between pairs (161).
- **H7 (agreement paradox): first part uninformative by construction; second part not significant.**
  - With the true N, the Chapman estimate is almost exactly N ÷ dependence ratio, so their Spearman correlation of −1.0 is an algebraic identity, not evidence. This was a design mistake in the pre-registration.
  - The empirical content of the "paradox" is H1 and the table above: higher agreement means a lower estimate, with intervals that stay narrow (correlation of dependence with interval width ρ = −0.28, p = 0.22).
- **H8 (dependence detectable without labels): supported, weakly.** The label-free log-linear model kept the most dependent pair (Haiku run 1 + run 2) as its *first* interaction. But it kept 13 of 21 interactions, so the test is weak.
- **H9 (keyword pairs least dependent, most accurate): supported, by a small margin.** Median dependence 1.115 vs 1.144; median error 10.4% vs 12.6%. Its main value is that it has different blind spots at zero cost.

## Limits
- **The lawyer labels are the truth here, and they miss clauses.** Checked by hand in the dry run, some "false positives" were valid clauses. Every estimate is of "clauses the lawyers labelled".
- **Labelling artefacts:** 13 labels are under 40 characters (headings, fragments). Removing them changes the median errors by under 1 point.
- **Single-pass agents on one task type (contracts).** No repeated sampling beyond the two Haiku runs.
- **The audit is simulated with the labels.** A real audit has its own error.
