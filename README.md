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
the **Web Search tool (Microsoft Web IQ)** to retrieve current web/news context, and
returns a **cited, house-style briefing**. Because the news / competitive /
regulatory / game-season landscape changes constantly, the input distribution
**drifts** — which is exactly what justifies the continuous-eval + auto-retrain
loop.

> **Note on WebIQ:** Microsoft **Web IQ** went GA at Build 2026 and is exposed in
> Foundry as the **Web Search tool**. We use it as the grounding path (verified
> working with GPT-5.4 on 2026-06-17). It needs no extra Bing resource and
> supports GPT-5-class models, unlike the now-deprecated classic "Grounding with
> Bing Search."

## What we measure (one simple number)

The whole demo tracks a single **Answer Quality Score** (0–100%): *is the
answer correct and supported by what it retrieved?* One line on the dashboard,
goes up as the model improves — deliberately legible to a non-data-science
audience. (A richer metric suite exists under the hood and matches the formal
agenda; we only demo the one number.)

## Datasets (evaluate on benchmarks, train on traces)

The core rule: **we evaluate on public benchmarks but never train on them** — the
student is distilled *only* on frontier + web-search (WebIQ) teacher traces, so the headline
metric stays honest. The benchmarks are layered by job:

| Layer | Dataset | Role |
| --- | --- | --- |
| Frozen regression | **`aialt/RetrievalQA`** (MIT) | "Does it still work?" — has a retrieval-needed flag, so grounding is *necessary*, not cosmetic |
| Temporal drift eval | **FreshQA** (dated snapshots) | "The world changed" — real drift across snapshots + false-premise traps |
| Replayable weekly stream *(optional)* | **RealTimeQA** (mirror) | Simulated incoming production queries over time |
| Hard ceiling | **`OpenResearcher/web-bench`** (Apache-2.0) | Frontier-vs-distilled headroom on hard web research |
| Training corpus | **Frontier + web-search (WebIQ) traces** | The *only* data the student learns from |

The frontier model answers questions *with web search (WebIQ) grounding* to produce the
gold-standard training traces. We distill **synthesis/citation over retrieved
context, not facts** — retrieval stays live, so the drift we retrain against is
*distribution* drift (new entities, false premises, query-mix shifts), not
fact-staleness. Details in
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
| Agent runtime | Foundry hosted agents + Web Search tool (Microsoft Web IQ) |
| Evals & tracing | Foundry tracing + continuous / batch eval |
| Model customization | **Primary:** GPU Managed Compute + LoRA/QLoRA on **qwen3-14b**. **Secondary (documented):** Foundry managed SFT on **qwen3-32b** |
| Golden dataset | Microsoft Fabric |
| Checkpoint storage | Azure Blob Storage |
| Eval results & completions | Azure SQL DB |
| Agent memory | Foundry IQ / Cosmos DB |
| Failure / latency / drift signals | Eventhouse |
| Dashboards | Power BI |

## Model customization — two paths

We show **both** customization modalities:

- **Primary (built): small OSS model on a GPU.** We fine-tune **qwen3-14b** with
  **LoRA/QLoRA** on a GPU **Managed Compute** endpoint — the "provision a GPU and
  code a PEFT loop" path. This is the genuinely-small open model Nikhil called
  for, and it gives us full control plus the clearest cost story.
- **Secondary (documented, not built): Foundry managed SFT.** Foundry offers a
  low-code **managed SFT** abstraction for **qwen3-32b** (Direct from Azure;
  SFT / DPO / RFT, Global training). It's the lowest-effort, most "cohesive
  Azure" path, but the base model is fine-tune-only (no base inference). We
  describe it here as the managed alternative rather than building it out.

Both stay within the **SFT / PEFT** guardrail (never full-parameter training).

## Status

Early design phase. These docs are drafts for the 6/26 internal session and are
meant to be reviewed with Nikhil (and Blu for the data/infra side).

## Constraints carried from prior project

- **No full fine-tuning.** Customization is limited to **distillation (SFT)** or
  **PEFT (LoRA / QLoRA)** — never full-parameter training.
- Keep correctness (eval quality) and serving performance (latency/cost) as
  **separate** measurements; don't conflate them.
