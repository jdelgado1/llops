# TMG Research Agent — LLMOps Lifecycle Demo

A reference implementation of the **full LLM lifecycle on Azure / Microsoft
Foundry**, framed around a **Telco, Media & Gaming (TMG) competitive- and
market-intelligence research agent**.

The thesis (per the 6/26 session agenda): **model choice is a lifecycle
decision** — quality, latency, cost, safety, drift, and retraining readiness.
We start with a frontier model, collect production traces, build a golden
dataset, run continuous evaluation, and **automatically retrain (via
distillation)** a smaller, cheaper, domain-specialized model that we promote
only when it passes quality / safety / latency / cost gates.

> **Start here:** [`CONTEXT.md`](CONTEXT.md) is the full handoff — decision
> history, locked direction, open questions, and a ready-to-paste prompt to
> continue the work. Read it before touching anything.

> **Why this demo exists:** prior work (function-calling fine-tune) already
> proved a 14B LoRA model can beat a frontier model on a narrow task. This repo
> reframes that proof as an **Azure-native, continuously-retraining LLMOps loop**
> around a realistic TMG research-agent workload.

## Scenario in one line

An internal **market & competitive intelligence research assistant** for a
Telco / Media / Gaming operator. Analysts and PMs ask grounded questions
("Summarize this week's competitor moves on 5G fixed wireless"), the agent uses
**Grounding with Bing Search** to retrieve current web/news context, and
returns a **cited, house-style briefing**. Because the news / competitive /
regulatory / game-season landscape changes constantly, the input distribution
**drifts** — which is exactly what justifies the continuous-eval + auto-retrain
loop.

> **Note on WebIQ:** the original agenda called for WebIQ, which is still in
> preview. We use **Grounding with Bing Search** (GA) as a drop-in substitute —
> it provides the same live-grounding / drift property and is available today.

## What we measure (one simple number)

The whole demo tracks a single **Answer Quality Score** (0–100%): *is the
answer correct and supported by what it retrieved?* One line on the dashboard,
goes up as the model improves — deliberately legible to a non-data-science
audience. (A richer metric suite exists under the hood and matches the formal
agenda; we only demo the one number.)

## Golden dataset (where the questions come from)

We **seed** the golden set from an existing Hugging Face Q&A dataset —
primary pick **`PatronusAI/financebench`** (analyst-style questions over real
company financials; telco/media/gaming are public companies), with
**`Maluuba/newsqa`** as a news-flavored alternative. The frontier model then
answers those questions *with Bing grounding* to produce the gold-standard
reference answers. Details in
[`docs/golden-dataset-and-eval.md`](docs/golden-dataset-and-eval.md).

## Documents

| Doc | Purpose | Owner / task |
| --- | --- | --- |
| [`docs/scenario.md`](docs/scenario.md) | Mock scenario: what the LLM does, who the users are | Jose — task #1 |
| [`docs/golden-dataset-and-eval.md`](docs/golden-dataset-and-eval.md) | Golden dataset schema + eval metrics (frontier vs distilled) | Jose — task #2 |
| [`docs/distillation-loop.md`](docs/distillation-loop.md) | Fine-tune/distillation loop design + Azure architecture | Jose — tasks #3–5 |

## Stack (target)

| Layer | Service |
| --- | --- |
| AI provider | Microsoft Foundry |
| Agent runtime | Foundry hosted agents + Grounding with Bing Search |
| Evals & tracing | Foundry tracing + continuous / batch eval |
| Model customization | Foundry distillation (SFT) — fallback: GPU + LoRA/QLoRA |
| Golden dataset | Microsoft Fabric |
| Checkpoint storage | Azure Blob Storage |
| Eval results & completions | Azure SQL DB |
| Agent memory | Foundry IQ / Cosmos DB |
| Failure / latency / drift signals | Eventhouse |
| Dashboards | Power BI |

## Status

Early design phase. These docs are drafts for the 6/26 internal session and are
meant to be reviewed with Nikhil (and Blu for the data/infra side).

## Constraints carried from prior project

- **No full fine-tuning.** Customization is limited to **distillation (SFT)** or
  **PEFT (LoRA / QLoRA)** — never full-parameter training.
- Keep correctness (eval quality) and serving performance (latency/cost) as
  **separate** measurements; don't conflate them.
