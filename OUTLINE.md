# Jev demo notebook: research notes and skeleton

Status: outline only, no code yet. Drafted 2026-10-09 from OpenRouter docs, TypeSafe docs,
OpenAI docs, and third-party evaluations published since Jev's 2026-09-15 launch.
Everything below post-dates my training data, so treat prices, model IDs and limits as
"verify at run time" (the notebook should fetch them live).

---

## 0. One-paragraph framing

Jev (TypeSafe, `typesafe/jev-1.13` on OpenRouter) is a "System One" decision model. You send
`state` (text/JSON) plus a map of typed `questions`; it returns one typed answer per question
with a probability distribution, never prose. Three primitives: `choice` (pick one of N
options), `noul` (P(yes)), `score` (position on an ordered rubric). Billed on input tokens only
($0.042/M, output free). The demo's thesis to test: Jev is ~10-20x cheaper and ~10x faster than
a GPT-6-class model for fixed-label classification, is within a few points of accuracy on
in-scope data, but confidently misfiles inputs that belong to no listed category, and an
explicit `other` option only partly fixes that. GPT-6 models have the same failure mode in the
"none of the above" literature, so the notebook should measure both.

---

## 1. Notebook skeleton

### Part A. Setup and key concepts
1. **Environment**: `OPENROUTER_API_KEY`; `requests` or the OpenRouter Python SDK
   (`alpha.decisions`). Optional `OPENAI_API_KEY` if we include OpenAI's Decisions API (see A.4).
2. **Discover decision models live**: `GET /api/v1/models?output_modalities=decisions`.
   Jev is hidden from the default model list. Print IDs, pricing, context length.
3. **First call**: the OpenRouter tutorial's support-ticket example (one `noul`, one `choice`,
   one `score` on the same state). Walk through the response shape:
   - `answers.<name>.noul` (0-1), `answers.<name>.choice` + `probabilities` + `confidence`,
     `answers.<name>.score` + `probabilities` + `legend`.
   - `model` returns the dated snapshot (`typesafe/jev-1.13-20260917`), `usage.cost` in USD.
4. **Endpoints**: `POST /api/alpha/decisions` (OpenRouter native) vs
   `POST /api/v1/systemone` (TypeSafe SDK compatible). OpenAI's Decisions API IS on OpenRouter
   as `openai/gpt-6-luna-decisions` ($0.10/M input, output free, 1.05M context, output
   modality `decisions`), so the whole comparison can stay on one OpenRouter key.
   (Verified 2026-10-09 via `GET /api/v1/models?output_modalities=decisions`.)
5. **What "System One" means and what Jev is not**: no reasoning trace, no explanation, no
   tools, no memory, text only, not a calculator. Restricted-softmax intuition: probability mass
   is normalised over only the declared options, so it always sums to 1 across your labels.

### Part B. How to prompt it (the "program" is the criteria)
1. **Anatomy of a question**: `type`, `instructions`, `criteria`. The question key is NOT sent
   to the model; option names and descriptions carry all the meaning.
2. **Criteria wording experiments** (small, cheap, visual):
   - Good vs vague vs wrong descriptions. Priorbench found wrong criteria scored 16.7%, below the
     25% random floor; missing descriptions cost only 0.8 points.
   - Option-order bias: reorder options, show how much the distribution moves (TypeSafe
     documents a first-option lean; priorbench saw up to 13 points on ambiguous items).
   - Structured criteria (`what` / `not_for` / `examples` objects) for confusable labels.
3. **Decomposition**: one broad question vs several atomic questions combined in code.
   Beri's phishing test: 62.6% as one noul, 95.0% as five atomic signals + logistic regression.
   Demonstrate on a small slice; this is the single biggest prompting lever.
4. **State hygiene**: send only the fields the decision needs. Accuracy drops with irrelevant
   state (TypeSafe's own jaggedness note). Show the same question with a lean vs bloated state.
5. **Batching**: independent questions in one request run in parallel; dependent logic needs
   separate calls. Up to 255 choice options, 10 score levels.
6. **Versioning**: pin `typesafe/jev-1.13` (thresholds are tuned per version), log the returned
   `model` string. `~typesafe/jev-latest` is an alias.
7. **The `other` option**: TypeSafe's docs recommend `other: "A ... that fits none of the above"`
   when the list may not be exhaustive. Set this up here; Part E tests whether it works.

### Part C. Reading the outputs
1. **Noul**: P(yes). Values near 0.5 are "unsure", not "medium severity". Run-to-run jitter
   (same input 0.52 then 0.49). Treat the middle band as its own outcome.
2. **Choice confidence**: TypeSafe's formula is `(p_max - 1/n) / (1 - 1/n)`. It summarises
   concentration, not correctness. OpenRouter's own Banking77 post says it is "not a calibrated
   probability" but ranks cases well (96.3% accurate at >=0.99 vs 29.6% below 0.5).
   Alternatives to compute ourselves: top probability, top-to-second margin, 1 - normalised entropy.
3. **Score**: probability-weighted index; show `legend`; warn that numeric interpolation is weak.
4. **Calibration plots**: reliability diagram + ECE against a bootstrap noise floor (small
   samples give non-zero ECE even for a perfect model). Compute from `probabilities`, not the
   undocumented `confidence` field. Third-party audits range from ECE 0.02 (CLINC150) to 0.09
   (Banking77) to 0.79 (unanswerable items with no abstain option).
5. **Three-band routing**: act / confirm / escalate, with thresholds picked on labeled data.
   Priorbench: accuracy flat from 0.50 to 0.95, 100% only at 0.99 covering 60% of traffic.

### Part D. Head-to-head vs GPT-6 class models (via OpenRouter)
1. **Models**: `openai/gpt-6-luna` ($0.10 in / $0.50 out), `openai/gpt-6-sol` ($2 / $10),
   optionally `openai/gpt-6-astra` ($10 / $50, subsample only). Prices from third-party
   trackers; fetch live.
2. **Fair prompting for the LLMs**: identical criteria text in the system prompt, utterance as
   user message, temperature 0, reasoning off, strict JSON schema with an enum of labels, prompt
   caching on the system message. Optionally also request `logprobs` for a probability-like
   signal if the provider exposes them.
3. **Metrics per model**: accuracy with bootstrap CI, macro-F1, per-class confusion, p50/p95
   latency (client-observed through OpenRouter), billed cost per 1,000 requests and per correct
   decision, invalid-output rate (LLMs can break schema; Jev cannot by construction).
4. **Cascade**: accept Jev when confidence >= t, else send to the LLM. Sweep t, plot
   accuracy vs cost. OpenRouter's Banking77 run: Jev alone 81.0% / $0.11 per 1k; Opus 84.4% /
   $2.42; cascade at 0.90 = 84.0% / $0.69 with 75.9% handled by Jev.
5. **Reference numbers to sanity-check against**: Banking77 (Jev 81.0 vs Opus 84.4);
   300-review synthetic set (Jev 96.1 vs Luna 97.1, Luna via chat completions); Rafe & Das
   preprint: zero-shot, no decision model beat Jev on intents, Gemma-4-31B beat it on CLINC-150
   at higher cost; fine-tuned DeBERTa-v3-large beat everything with labels.

### Part E. The unknown-category problem (centrepiece)
1. **Mechanism**: restricted softmax forces every input into a declared label. Show a cake
   recipe sent to a support-ticket router (priorbench saw `technical` at 0.94; random letters at 0.97).
2. **Experiment 1, no `other` option**: in-scope accuracy, OOS "flag rate" (should be 0 by
   construction), and whether confidence/probability separates OOS from in-scope (AUROC of
   max-prob as an OOS detector). Priorbench: 0 of 30 OOS flagged even at 0.99 confidence.
3. **Experiment 2, with `other`**: same data, add `other`. Measure OOS recall into `other`,
   false `other` on in-scope, and how many OOS items are still confidently assigned to a real
   label. Prior evidence that it helps but does not fix it:
   - calibration-probe: Jev confident on 47% of unanswerable items without an abstain option;
     with it, 77% abstain but 13% still answer confidently (Claude Sonnet 5 baseline: 20% -> 3%).
   - Rafe & Das: an in-scope threshold still admitted 31% of out-of-scope requests.
   - scienthoon OOD audit: the "unknowable" label averaged 0.74 probability.
   - jujumilk3 KoBBQ: removing the unknown option took accuracy on unanswerable items from
     0.95 to 0.00 and ECE from 0.023 to 0.793.
4. **Experiment 3, near vs far OOS**: far OOS (recipes, code, gibberish) vs in-domain OOS
   (banking queries whose intent we deliberately held out). The 2021 "Are Pretrained Transformers
   Robust in Intent Classification" paper shows far-OOS is much easier and over-states robustness.
5. **Experiment 4, same tests on GPT-6 Luna/Sol**: the LLM literature predicts a similar
   failure. Tam et al. (arXiv 2503.01550, 28 models): accuracy drops 30-50% when "none of the
   above" is correct. arXiv 2402.01349: models pick the "least incorrect" option. Compare
   whether Jev or GPT-6 handles `other` better, and whether an abstain threshold on Jev's
   probability beats the LLM's explicit `other`.
6. **Mitigations to show**: (a) always include `other` with a concrete description;
   (b) a separate relevance `noul` ("Is this about banking?") run first; (c) threshold on
   max-prob or margin, tuned on held-out data; (d) hierarchical choice (domain first, then intent).

### Part F. Costs
1. Pull `usage.cost` from every call and build a cost table: per request, per 1k, per correct
   decision, projected per million. Jev input tokens scale with option count (77 Banking77
   options averaged 2,605 input tokens).
2. Compare to LLM costs with and without prompt caching (Opus without caching was ~$19 per 1k
   on Banking77 vs $2.42 with).
3. Budget for the full notebook (rough, at list prices, 1,500 utterances x ~3k tokens):
   Jev ~ $0.20; Luna ~ $0.50 + output; Sol ~ $10; Astra ~ $50. Astra only on a subsample.

### Part G. Limitations (TypeSafe's own "jaggedness" list plus third-party findings)
- Literal reading, counting, numeric representations, score arithmetic, dates, double negatives,
  large/irrelevant state, adversarial content ("does not treat it as hostile by default"),
  contradictory instructions, option-order lean, no generation, bounded context (32k on
  OpenRouter; TypeSafe says 64k total with 32k for state).
- Prompt injection results are mixed: one bench flipped 1 of 1,056 attacks, another saw
  accuracy fall 96.5% -> 26.5% with a one-line injected instruction. Worth a tiny demo.
- Non-English: Russian 77% vs English 88% in one audit; CJK lower per TypeSafe.
- Negation consistency: P(x) + P(not x) spans 0.93-1.19 across pairs.
- Evidence quality caveat for the notebook text: nearly all third-party numbers are from
  GitHub repos and preprints published in the last three weeks, single runs, often synthetic
  data. TypeSafe's own headline accuracy is agreement with two frontier LLMs, not ground truth.

### Part H. Wrap-up
- Decision matrix: when Jev, when an LLM, when a fine-tuned encoder, when a regex.
- Shadow-eval recipe for a real deployment (pin version, include `other`, calibrate per
  question on half the data, test injection, route only the confident band).

---

## 2. Data

| Need | Candidate | Why | Notes |
|---|---|---|---|
| In-scope intent classification with a real OOS split | **CLINC150** (Larson et al. 2019, EMNLP) | 150 intents, 10 domains, 1,000 OOS test items shipped with the dataset. The standard OOS benchmark. Also used by Rafe & Das, so numbers are comparable. | Verified from `clinc/oos-eval` `data_full.json`: train 15,000 (100/intent), val 3,000 (20/intent), test 4,500 (30/intent) + OOS 100/100/1,000. Rows are `[text, label]` pairs; median 8 words. HF name is `clinc/clinc_oos` (configs `plus`, `small`, `imbalanced`). 150 options per choice question fits the 255 limit. |
| Direct comparability with OpenRouter's published Jev numbers | **Banking77** (PolyAI, CC BY 4.0) | 77 intents, 3,080 test utterances; OpenRouter already ran Jev vs Opus on it. | Verified from PolyAI's repo CSVs (`text`, `category`): train 10,003 (35 to 187 per intent, imbalanced), test 3,080 (40 per intent); median 9 to 10 words. No OOS class. Build in-domain OOS by holding out N intents from the option list. Possible LLM memorisation (2020 dataset). HF name `legacy-datasets/banking77`. |
| Far-OOS probes | Small hand-built set (~50): recipes, code snippets, lorem ipsum, other-domain queries | Reproduces the priorbench "cake recipe" failure vividly. | Cheap; write it in the notebook. |
| Prompting/decomposition demo | ~200 phishing emails or the CLINC subset | To show atomic-question decomposition. | Beri's set (PhishNChips) had synthetic bodies and proxy labels; CLINC is cleaner. |

Recommendation: CLINC150 as the primary dataset (it answers the OOS question directly), a
Banking77 subsample for cross-checking against OpenRouter's post, plus the hand-built far-OOS set.
Subsample to ~1,000-1,500 in-scope + 300-500 OOS to keep GPT-6 Sol under ~$10.

---

## 3. Other models to explore

**Decision models on OpenRouter's Decisions API** (same three primitives; list and verify with
`GET /api/v1/models?output_modalities=decisions`):

| Model ID | Notes |
|---|---|
| `typesafe/jev-1.13`, `~typesafe/jev-latest` | The subject. $0.042/M input. |
| `typesafe/jev-router` | Different thing: a chat-completions model router powered by Jev. Could be a sidebar, not a comparison. |
| `inception/mercury-decide:free` | Free in early access since 2026-09-30; vendor claims top of JevBench. Good zero-cost second decision model. |
| `jaredpalmer/kev-4b` | Open-weight (Apache-2.0) LoRA + pointer head on Qwen3.5-4B; $0.042/M. Lets us say "you can self-host this class of model". |
| `perplexity/pplx-decider-v1-27b` | Apache-2.0 weights, $0.02/M input per benchlm. |
| `upstage/solar-decide`, `togethercomputer/tev1-4b-experimental`, `liquid/d1`, `respan/span-01(-lite)`, `cloudflare/clef(-flash)` | Seen in OpenRouter usage stats; pricing unverified. Pick one or two, not all. |

**OpenAI Decisions API, on OpenRouter as `openai/gpt-6-luna-decisions`**: $0.10/M input only,
1.05M context. OpenAI's native primitives are `predicate` / `choice` / `score` with a `refusal`
answer type and inline base64 image support; confirm in the notebook which of those OpenRouter's
`/api/alpha/decisions` wrapper exposes (it may normalise to `noul`/`choice`/`score`). OpenAI's
guide itself recommends an `other` fallback option. This is the most direct apples-to-apples
competitor; plus `openai/gpt-6-luna` via chat completions shows the same model with and without
the decision head.

**Live decision-model catalog on OpenRouter (2026-10-09, 19 entries, all input-only billing,
prices per 1M input tokens):**

| Model ID | $/M in | Context | Notes |
|---|---|---|---|
| `typesafe/jev-1.13`, `~typesafe/jev-latest` | 0.042 | 64,000 | Catalog says 64k, docs said 32k |
| `openai/gpt-6-luna-decisions` | 0.10 | 1,050,000 | OpenAI Decisions API |
| `inception/mercury-decide`, `:free` | 0.02 / 0 | 65,536 | Free tier for the demo |
| `perplexity/pplx-decider-v1.1-27b` | 0.02 | 262,144 | Apache-2.0 weights |
| `jaredpalmer/kev-4b` | 0.042 | 8,192 | Open weights, small context |
| `microsoft/microsoft-decision-1` | 0.042 | 32,768 | |
| `nace-ai/drex-v1.5` | 0.04 | 131,072 | |
| `upstage/solar-decide`, `-flash` | 0.05 | 524,288 | |
| `cloudflare/clef`, `clef-flash`, `clef-omni` | 0.24 / 0.038 / 0.15 | 65,536 | |
| `liquid/d1` | 0.04 | 65,536 | |
| `togethercomputer/tev1-4b-experimental` | 0.042 | 32,768 | |
| `respan/span-01`, `-lite`, `-lite:free` | 0.02 / 0 / 0 | unlisted | |

**Open baselines worth one cell each**: a fine-tuned encoder (DeBERTa/ModernBERT) if labels
exist, which beat every decision model in Rafe & Das; TF-IDF + logistic regression as the
floor. Leaderboard context: benchlm.ai JevBench v1.5.4 ranks 106 systems (Jev 1.13 third behind
two open Gemma/12B-class models), but that is a composite index, not accuracy.

---

## 4. Open questions before writing code

1. **OpenAI Decisions API**: now confirmed on OpenRouter, so the question is only whether to
   include both `gpt-6-luna-decisions` and chat-completions `gpt-6-luna` (recommended: yes, both).
2. **Which GPT-6 tiers**: Luna only (cheap, closest competitor), Luna + Sol, or all three with
   Astra on a subsample? Drives budget (Part F.3).
3. **Dataset**: CLINC150 primary as recommended, or Banking77 to mirror OpenRouter's post?
4. **Budget ceiling** for API spend in the notebook.
5. **Audience**: engineers evaluating adoption (lean on Part E and cascade economics) or a
   broader demo (lean on Parts A-C)? Affects depth and how much of the robustness literature to cite.
6. **Second decision model**: Mercury Decide (free) and/or Kev-4B (open weights), or Jev alone?

---

## 5. Sources consulted

Official: OpenRouter Jev docs and tutorial (openrouter.ai/docs/guides/community/jev,
/jev-tutorial), OpenRouter "What is Jev" and "Is Jev as accurate as frontier models"
(Banking77) posts, OpenRouter decisions skill page, TypeSafe docs (introduction, confidence,
primitives/choice, state, model-jaggedness/jev-1.13, llms.txt index), OpenAI Decisions guide
(developers.openai.com/api/docs/guides/decisions), Vercel AI Gateway gpt-6-luna-decisions page.

Third-party evaluations: priorbench/jev (pre-registered, 5,721 calls), beri.net write-up of
the phishing and OOD studies, Hyperstack restricted-softmax explainer,
andre-langchain/calibration-probe, jujumilk3/jev-calibration-audit, Rafe & Das arXiv:2610.00346,
Open-Jev/JevLite arXiv:2609.23959, mameli/jev-vs-luna, Yifan-Lan/awesome-jev-robustness
(index of ~146 studies), benchlm.ai/decision-models.

LLM "none of the above" literature: Tam et al. arXiv:2503.01550; arXiv:2402.01349
("least incorrect option"); arXiv:2409.00113; Larson et al. 2019 (CLINC150);
arXiv:2106.04564 (OOS evaluation gaps in Banking77).

Could not fetch directly (403/404): openrouter.ai/typesafe/jev-1.13 model page, openai.com
pricing page, JevBench paper on ResearchGate. Prices for GPT-6 tiers came from llmreference.com
and computeprices.com and should be re-fetched live in the notebook.
