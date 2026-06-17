# Mock Scenario — TMG Market & Competitive Intelligence Research Agent

> Task #1 from Nikhil's list: *Define mock scenario — what are they using the
> deployed LLM for, who are the users.*

## Client Archetype

A large **Telco, Media & Gaming (TMG)** operator — a company spanning some or all
of:

- **Telco:** broadband / mobile / 5G fixed wireless access (FWA), MVNO partners.
- **Media:** OTT streaming (SVOD/AVOD), content licensing, ad-supported tiers.
- **Gaming:** live-service / free-to-play titles, battle passes, in-game economies.

(The same scenario works for a **consultancy or account team serving TMG
clients** — the agent becomes their research surface for client briefings.)

## What the Deployed LLM Does

It powers an internal **Market & Competitive Intelligence research assistant**.
Given a natural-language question, the agent:

1. Plans the query and calls **Grounding with Bing Search** to pull current
   web / news / analyst commentary.
2. Optionally calls internal tools (function calling): a product-spec knowledge
   base, a pricing/plan lookup, an internal metrics service.
3. Synthesizes a **cited, house-style briefing** — concise, sourced, and written
   in the company's analyst voice.

Function calling is therefore a **capability inside the agent**, not the headline
task — which lets us reuse prior function-calling expertise while keeping the
grounded-research drift story front and center.

## Who the Users Are

| Persona | Typical need |
| --- | --- |
| Competitive intelligence analyst | "What did competitors announce this week?" |
| Product manager (broadband / streaming / live-service) | Feature & pricing benchmarking |
| Go-to-market / marketing | Campaign and positioning context |
| Strategy / corp dev | Earnings, regulatory, and market-trend briefings |
| Executive | Short, sourced briefings before meetings |

## Representative Queries

- "Summarize this week's competitor announcements on **5G fixed wireless access**
  in the US, with sources."
- "What **EU regulatory changes** this quarter affect **OTT streaming bundles**?"
- "Compare **player sentiment** on our latest **battle pass** vs the competitor's
  new season launch."
- "Give me a briefing on **ARPU and churn** trends from the top 3 carriers' last
  earnings calls."
- "What are analysts saying about **ad-supported streaming tiers** adoption this
  month?"

## Concrete Golden-Set Seed (for the Demo)

For the actual demo we seed the questions from an existing Hugging Face Q&A
dataset rather than hand-writing them — primary pick **`PatronusAI/financebench`**
(analyst-style questions over real company financial filings; telco/media/gaming
are all public companies), with **`Maluuba/newsqa`** as a news-flavored
alternative. This keeps the scenario realistic *and* reproducible. See
[`golden-dataset-and-eval.md`](golden-dataset-and-eval.md) for how to pick and
use it.

## Why This Is a Strong Drift Scenario

News, competitor moves, regulations, earnings, and game-season sentiment change
**continuously**. The input distribution *and* the "correct" grounded answer
shift week-to-week. A model that scores well in Q1 quietly degrades by Q3 as:

- new competitor products / terminology appear,
- the retrieved Bing context distribution changes,
- query mix shifts (e.g., a major game launch floods gaming questions).

This is precisely the condition that justifies **continuous evaluation +
automatic retraining**, the centerpiece of the demo.

## Why Customization (Distillation) Pays Off Here

The task is **narrow and domain-heavy**:

- Dense TMG jargon: MVNO, ARPU, churn, FWA, OTT, SVOD/AVOD, DAU/MAU,
  live-service, battle pass, cohort retention, spectrum, bundle, cord-cutting.
- A consistent **house output style**: short, sourced, structured briefings.

A smaller model **distilled** from the frontier model can internalize the domain
conventions and house style, matching frontier quality **on this narrow task** at
materially lower **cost and latency** — the payoff the demo argues for.

## Success Criteria (Scenario-Level)

The demo "works" if we can show, end-to-end:

1. A frontier baseline producing high-quality grounded briefings + traces.
2. A golden dataset built from those traces (see
   [`golden-dataset-and-eval.md`](golden-dataset-and-eval.md)).
3. Continuous eval detecting a **drift-induced quality drop**.
4. An **automatic distillation run** producing a smaller candidate.
5. A **promotion gate** that ships the candidate only when its **Answer Quality
   Score** (plus safety, latency, and cost) clears the bar.

## Open Questions for the Team

- Single TMG segment for the demo (e.g., telco only) for tighter scope, or all
  three to show breadth? Recommendation: **build all three into the golden set,
  demo with telco + gaming** for variety.
- Do we want internal "private" tools (KB / pricing) mocked, or Bing-only for
  v1? Recommendation: **Bing-only for v1**, add one mocked internal tool if time.
- Which frontier model as teacher — Claude Opus or GPT-5.5-class? (Affects trace
  collection setup.)
