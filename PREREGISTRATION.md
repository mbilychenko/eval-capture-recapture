# Pre-registration

*Written 2026-10-18, before any agent was run on the full sample. Committed to git before the full run, so the timestamp shows what was decided in advance. Changes after that point are listed at the bottom, with dates and reasons. Results are reported whichever way they go.*

## 1. Question
Several AI agents review the same documents, and nobody has labelled everything. Can capture–recapture estimate:
- how many items **all** the agents missed;
- each agent's **recall**?

And when does it go wrong? The forest example (Act 1 of the article) named four ways a two-source estimate fails. Each is tested here, on data where the truth is known:

| Mechanism in the forests | Analogue for AI agents | Expected bias | Tested by |
|---|---|---|---|
| Famous forests are in both maps | Some clauses are obvious to every model | too low | H3 |
| Maps copied from each other | Same model, same prompt, same family | too low | H2 |
| Maps that specialise | Agents that focus on different clause types | too high | H4 |
| The same forest not linked across maps | Loose or strict matching of quotes | depends on the rule | H5 |

## 2. Data (frozen)
- **Dataset:** CUAD v1, 510 commercial contracts with clauses labelled by lawyers (The Atticus Project, CC BY 4.0).
- **Sample:**
  - all 294 qualifying contracts (seed `20261018` fixes the order);
  - only contracts of at most 40,000 characters (294 of 510 qualify);
  - drawn by `scripts/01_sample.py`.
- **10 clause categories:** chosen before running any model, for being open-ended:
  - License Grant, Cap On Liability, Anti-Assignment, Audit Rights, Insurance;
  - Post-Termination Services, Minimum Commitment, Revenue/Profit Sharing, Exclusivity, Ip Ownership Assignment.
- **Unit:** one labelled clause (contract, category, span); overlapping labels are merged.
- **True N = 1,413**, from 75 to 238 per category.
- **The lawyer labels are treated as the truth.** Clauses the lawyers missed but an agent found count as false positives here. This is reported as a limit.

## 3. Agents

**Main set (7 agents, a ladder from "copies" to "strangers"):**

| Agent | Model | Prompt | Rung |
|---|---|---|---|
| haiku_run1 | Claude Haiku 5.5 | standard | — |
| haiku_run2 | Claude Haiku 5.5 | standard | same model, same prompt, second run |
| haiku_alt | Claude Haiku 5.5 | checklist (different wording and procedure) | same model, different prompt |
| sonnet | Claude Sonnet 5.5 | standard | same family, larger model |
| gpt_luna | OpenAI GPT-6 luna | standard | other family, small |
| gpt_sol | OpenAI GPT-6.1 sol | standard | other family, larger |
| keyword | keyword rules, no AI (`src/crc/keyword_agent.py`) | none | different method entirely |

- The design is a 2×2 at matched prices: Haiku 5.5 and GPT-6 luna both cost $0.10 / $0.50 per million tokens, and Sonnet 5.5 and GPT-6.1 sol both cost $2 / $10. A difference between families can't come from one side buying the stronger model.
- **Specialists**, used only for H4: `haiku_specA` and `haiku_specB`. They see all 10 clause types, but each is told to focus on a different half and report the other half only when it's obvious.
- All runs use temperature 1.0. OpenAI reasoning effort is `low`; no Claude extended thinking.
- Each agent reads a whole contract once and quotes every clause of the requested types. These are single-pass agents; the method doesn't depend on how an agent works inside.
- Model IDs and API parameters are confirmed at the dry run (5 contracts). A change goes into §9.

## 4. Matching (frozen)
- **Locating a quote:** exact match (whitespace and case ignored), or else fuzzy alignment with a score of at least 90. Quotes that can't be located catch nothing, and their share is reported per agent.
- **Labelled setting:** a quote catches a labelled clause if their overlap is at least **0.5** of the shorter of the two. 0.3 and 0.7 are the sensitivity check (H5).
- **Label-free setting:** quotes from all agents in the same contract and category are joined into one item when they overlap by the same rule. Each joined item is one "catch" for every agent that quoted it. False positives are included, as a practitioner would see them.

## 5. Estimators
- **Two sources:** Chapman, with a 95% interval.
- **All main agents:**
  - Chao Mh;
  - first- and second-order jackknife (Burnham & Overton), designed for items that differ in how easy they are to catch;
  - a log-linear model with pairwise dependence chosen by AIC (Fienberg).
- **Repeated trials:** every estimate is made separately for each of the 10 categories, so each category is its own "lake" with a known N. That gives 21 pairs × 10 categories = 210 checks against the truth. Intervals for the totals come from resampling contracts (1,000 bootstrap resamples).
- **Recall estimate** for agent *i*: r̂ᵢ = (items found by *i*) ÷ N̂, against the true recall rᵢ = (labelled clauses caught by *i*) ÷ 1,413.
- **Common-practice baseline: pooled recall.** This treats everything any agent found as the full truth, as many evals do: rᵢ ≈ (found by *i*) ÷ (found by anyone).
- **Audit correction (label-free setting):**
  - 10% of the joined items (seeded) are "checked by a human", simulated with the labels;
  - from the audit, the share of real clauses is estimated separately for items caught by 1, 2, … agents;
  - the frequency counts are scaled by those shares before estimating.
- **Dependence of a pair:**
  - the dependence ratio = observed overlap ÷ overlap expected under independence (uses the true N);
  - the **excess dependence** = a pair's ratio ÷ the median ratio of the 8 Anthropic × OpenAI pairs.

## 6. Primary outcome
**How accurately each main agent's recall can be estimated without labels.** Measured as the mean absolute error of r̂ᵢ against the true rᵢ across the 7 main agents. It is computed three ways:
- (a) pooled recall;
- (b) the best two-source estimate (the most diverse pair, chosen by the H8 diagnostic, so without labels);
- (c) the all-agent estimate with the audit correction.

**Primary hypothesis H0:** (c) has a smaller error than (a).

## 7. Secondary hypotheses
- **H1, strangers work:** for cross-family pairs, the Chapman interval contains the true per-category N in at least half of the (pair × category) lakes, labelled setting.
- **H2, copied maps:** the median relative error (N̂ ÷ N − 1) is lowest, meaning furthest below the truth, for the same-model pair, then same model with a different prompt, then same family, then cross-family.
- **H3, famous forests:** even cross-family pairs underestimate (median relative error below 0) because clauses differ in how obvious they are. The heterogeneity estimators (jackknife, Chao) reduce that gap.
- **H4, specialising maps:** the specialist pair (specA, specB) overestimates N (median relative error above 0).
- **H5, matching:** in the label-free setting, the estimated N changes monotonically with the matching threshold. The size of the effect is reported against the two-source spread.
- **H6, more sources:** the all-main-agent log-linear and second-order jackknife estimates have a smaller absolute error than the median two-source estimate.
- **H7, the agreement paradox:** across the 21 main pairs:
  - higher agreement goes with a lower estimate (Spearman correlation between dependence ratio and N̂ ÷ N below 0);
  - higher agreement also goes with a narrower interval (Spearman correlation between dependence ratio and interval width ÷ N̂ below 0).

  Agreement makes the estimate look *more* certain while it moves *further* from the truth.
- **H8, finding blind spots without labels:** the pairwise interactions the log-linear model selects in the label-free setting (no truth used) include the pair with the highest true dependence ratio among the main agents.

- **H9, a different method:** the 6 pairs that include the keyword checker have the lowest dependence ratios of all pairs (median below the median of the 15 LLM–LLM pairs), and their median absolute error in N is below that of the LLM–LLM pairs.

## 8. Exclusions
- A contract is dropped for all agents if any agent fails on it after 3 retries. The count is reported.
- If the dry run shows that an agent's output can't be parsed or located for more than 20% of its quotes, its prompt or parameters may be fixed **before** the full run. Any such fix is logged in §9.

## 9. Changes after registration
- **2026-10-18, dry run.** The Anthropic SDK in use (1.15) no longer accepts a `temperature` parameter, so Claude agents run with the API's default sampling; OpenAI agents keep temperature 1.0. Repeated runs of the same Claude model (haiku_run1 vs haiku_run2) still differ through sampling, which the dry run checks.
- **2026-10-18, before the full run (dry run: 5 contracts, no hypothesis computed).**
  - Added a rule-based keyword checker as a 7th main agent (and H9). It is the most independent net available, and readers can rerun it without API keys. Its rules were written from general contract vocabulary, and were not tuned on the labels.
  - Widened the sample from 150 contracts to all 294 under the length cap (true N 690 → 1,413), so the per-category checks have more clauses. The extra cost is about $4.
  - Reason for both: the dry run suggested strong dependence between all the LLM agents, so the design needs a more independent net and tighter intervals to tell the outcomes apart.
