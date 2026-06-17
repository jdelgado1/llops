# Project Context & Handoff

> **Read this first.** This file is the single source of truth for the TMG
> Research Agent LLMOps demo. It captures *what we're building, why, every
> decision we made (and reversed), and how to continue.* If you're an AI agent
> or a teammate picking this up cold, read this top-to-bottom before touching
> anything else.

Last updated: 2026-06-17 · Owner: **Jose Delgado**

---

## 1. The One-Paragraph Summary

We're building a **demo of the full LLM lifecycle on Azure / Microsoft Foundry**
for an internal session on **6/26**. The story: **"model choice is a lifecycle
decision."** Start with an expensive **frontier model** (Claude Opus / GPT-5.5),
use it to produce gold-standard answers, then **distill a cheaper, smaller model**
that matches its quality on a narrow task — and **continuously retrain** it as the
world drifts. The task is a **Telco/Media/Gaming (TMG) grounded research agent**
that answers questions using **live web search (Bing grounding)**. Success is
measured by **one simple number** (an "Answer Quality Score") tracked over time.

---

## 2. Who's Who

- **Jose (you):** owns the scenario, golden dataset + eval metric, and the
  fine-tune/distillation loop. (This repo.)
- **Nikhil Gopal:** lead / presenter. Set the direction. Wants **simple,
  legible** concepts for a mixed infra/data audience.
- **Blu Gotlieb:** owns the database, Event Hub, and dashboards (drift &
  performance over time, likely Power BI).
- **Anthony Nevico:** stakeholder; gives feedback before the session.

---

## 3. What We're Actually Doing (Plain English)

1. Take **one Hugging Face Q&A dataset** (questions + correct answers).
2. Split it: a **training pile** and a hidden **golden test pile**.
3. Have the **frontier model** answer the questions (with Bing grounding) →
   these are the gold-standard reference answers.
4. **Score the frontier model** on the golden pile → that's the quality bar
   (one number, e.g. 94%).
5. **Score a cheap model** → it does worse (e.g. 78%).
6. **Distill / fine-tune the cheap model** on the training pile → it gets close
   to the frontier model (e.g. 92%) but far cheaper + faster. **That's the win.**
7. **Schedule automatic retraining** and **store every score in a database**
   (with Blu) so we can chart quality over time.
8. **Drift story:** as the world changes (news, new topics), the model's score
   drops → auto-retrain brings it back up. That's *why* you keep retraining.

> The metric is the scoreboard: **one accuracy/quality percentage** that goes up
> as the model improves. Keep it that simple.

---

## 4. Key Decisions (and Why)

| # | Decision | Why | Status |
| --- | --- | --- | --- |
| D1 | **New repo**, separate from the old Nebius function-calling repo | The old repo is a different stack (Nebius/SGLang/Axolotl) and narrative; a clean Azure/Foundry repo tells a clean story | ✅ Locked |
| D2 | **Distillation** as the primary customization method (not from-scratch fine-tune) | The scenario already produces frontier "teacher" outputs; Foundry has native distillation; less labeling than DPO/RFT; on-message for "cohesive Azure" | ✅ Primary (DPO/RFT mentioned as alternatives in the talk) |
| D3 | **No full fine-tuning, ever** | Carried constraint from prior project. Only **distillation (SFT)** or **PEFT (LoRA/QLoRA)** | ✅ Hard rule |
| D4 | Scenario = **TMG grounded research agent** (Telco/Media/Gaming) | Nikhil wanted a TMG-relevant customer use case; research + live grounding gives the strong drift story | ✅ Locked |
| D5 | **Web search (Bing grounding)**, not classification, not tool-calling, as the task | Matches Nikhil's *written agenda* (WebIQ research agent). WebIQ is in preview → **Bing grounding is the GA fallback** Nikhil named | ✅ Locked |
| D6 | **One simple headline metric** ("Answer Quality Score") | Nikhil's *verbal* guidance: audience is infra/data people, "simple is better." Richer metrics exist under the hood but we demo one number | ✅ Locked |
| D7 | Golden set = **HF Q&A dataset (seed) + frontier traces (reference)** | Nikhil said HF has representative datasets; seeding gives reproducibility; frontier answers give the gold standard | ✅ Locked |
| D8 | HF dataset = **`PatronusAI/financebench`** (primary), `Maluuba/newsqa` (alt) | FinanceBench = analyst QA over real company financials (TMG companies are public); has reference answers + cited evidence; credible benchmark | 🔶 Proposed — confirm dataset card |

### Decisions we REVERSED (don't re-litigate)

- ❌ **Classification task** (e.g., support-intent classification on Banking77).
  Briefly proposed because Nikhil's *verbal* note praised simple accuracy
  metrics. **Reversed** — his *written agenda* clearly centers a **web-search
  research agent**. We kept the "simple metric" idea but applied it to the
  research task (one quality score). The word "classification" should not
  reappear as the task.
- ❌ **Pure trace-built golden set with no HF dataset.** Replaced by the
  **HF-seed + traces** hybrid (D7).
- ❌ **Groundedness/citation multi-metric suite as the headline.** Too
  data-science-y for the audience; demoted to "full version under the hood"
  behind the single Answer Quality Score (D6).

---

## 5. The Source-of-Truth Context (from Nikhil)

### 5a. Session agenda (Nikhil's written draft)

| Time | Segment | Purpose |
| --- | --- | --- |
| 0–5 | Goal framing | Model choice is a lifecycle decision: quality, latency, cost, safety, drift, retraining readiness |
| 5–10 | Start with a frontier model | Claude Opus / GPT-5.5-class baseline; collect traces |
| 10–18 | Use case: WebIQ research agent | Live web/news grounding → input distribution drifts over time |
| 18–30 | Golden dataset creation | Capture prompts, retrieval context, completions, citations, human ratings, failure labels, latency, token cost |
| 30–42 | LLMOps evaluation pipeline | Continuous + batch evals in Foundry: groundedness, task adherence, fluency, safety, citation quality, latency, cost |
| 42–52 | Automatic retraining / customization | Foundry: SFT/distillation, DPO, or RFT with graders |
| 52–58 | Architecture walkthrough | Foundry hosted agents, Fabric storage, checkpoints, eval results, drift monitoring |
| 58–60 | Decisions & next steps | Trace capture, golden set, eval gates, retraining trigger, promotion criteria |

### 5b. Nikhil's verbal guidance (paraphrased from transcript)

- Find a **representative golden dataset** for a **hypothetical customer
  scenario** where they'd generate lots of production data with a big frontier
  model (Opus / GPT-5.5).
- **Hugging Face** has chatbot/QA datasets that could be representative.
- Define a **metric to quantify the model's progress** over time. Examples:
  *"how often does it use the right tool"* or *"% of correct category"* — or an
  **amalgamation of metrics combined into one coefficient.**
- **Simple is better** — audience has infra and data people who aren't
  LLM-native.
- Flow: golden set → eval frontier vs cheap model → **fine-tune the cheap model
  and make it good** → schedule automatic training jobs → store eval results in
  a DB with Blu → track progress over time.
- **Why retrain:** users constantly web-searching → input drifts. WebIQ is a
  great example of this.

### 5c. Important caveats Nikhil gave

- He **applied for WebIQ + RL-environment preview access**. If not approved,
  **fall back to normal Bing grounding** + another training method. → We default
  to **Bing grounding** now.
- *"For the purposes of this demo the exact implementation details are less
  crucial than the concepts."*
- *"We are by no means married to this agenda."* → The plan is flexible; concepts
  win over specifics.

### 5d. Target architecture (Nikhil's)

| Layer | Service |
| --- | --- |
| AI provider | Microsoft Foundry |
| Agent runtime | Foundry Hosted Agents (+ Grounding with Bing Search) |
| Evals & tracing | Foundry Tracing |
| Checkpoint storage | Azure Blob Storage |
| Eval results & prompt/completions | Azure SQL DB |
| Persistent agent memory | Foundry IQ / Cosmos DB |
| Failures / latency / drift signals | Eventhouse |
| Golden dataset | Microsoft Fabric |
| Dashboards | Power BI |

---

## 6. Current State of This Repo

```
tmg-research-agent-llmops/
├── README.md                         # Overview, thesis, stack, one-number metric, HF seed dataset
├── CONTEXT.md                        # ← this file
└── docs/
    ├── scenario.md                   # Task #1 — the mock TMG research-agent scenario
    ├── golden-dataset-and-eval.md    # Task #2 — HF-seed + traces golden set, single quality metric, dataset-picking guide
    └── distillation-loop.md          # Tasks #3–5 — distillation loop + Azure architecture
```

Everything is **design docs only** — no code yet. All three docs reflect the
locked decisions above (research agent + Bing + one metric + HF seed).

---

## 7. Jose's Assigned Tasks (Nikhil's "By Wed" list)

1. ✅ (drafted) Define the **mock scenario** — `docs/scenario.md`.
2. ✅ (drafted) Identify the **golden dataset + eval metric** —
   `docs/golden-dataset-and-eval.md`.
3. ⬜ Identify **data to distill on** (HF dataset + frontier completions).
4. ⬜ Build the **distillation loop** (Foundry-native; fallback GPU + LoRA/QLoRA).
5. ⬜ **Schedule automatic retraining** + store eval results in DB (with Blu).

---

## 8. Open Questions to Resolve Next

- [ ] Confirm **`PatronusAI/financebench`** fits (columns, license, splits) — or
      pick a different HF Q&A dataset.
- [ ] Choose the **frontier teacher** model (Claude Opus vs GPT-5.5-class).
- [ ] Choose the **student** model (small, Foundry-deployable, PEFT-friendly).
- [ ] Define the exact **Answer Quality Score** (groundedness %? LLM-judge
      correctness %? blended coefficient?) and the **promotion gate** thresholds.
- [ ] Decide how to **simulate drift** with a static dataset (temporal split /
      inject new topics / shift distribution).
- [ ] Confirm Foundry distillation stays within **SFT/PEFT** (not full-param).
- [ ] Align with **Blu** on the DB schema for storing eval results over time.

---

## 9. Hard Rules / Guardrails

- **No full fine-tuning.** Distillation (SFT) or PEFT (LoRA/QLoRA) only.
- **Task = grounded research via Bing**, not classification, not tool-calling.
- **Demo one simple metric.** Don't lead with a multi-metric dashboard.
- **Keep quality (accuracy) and performance (latency/cost) as separate
  numbers.** Don't conflate.
- Concepts > implementation details (Nikhil's words). Don't over-engineer.

---

## 10. Prompt to Continue This Work

Paste the following into a new chat (with this repo open) to pick up exactly
where we left off:

```text
You are helping me (Jose) build an Azure / Microsoft Foundry demo of the full
LLM lifecycle for an internal session on 6/26. Before doing anything, read
CONTEXT.md in this repo end-to-end — it has the full decision history, the
locked direction, and open questions. Treat its "Key Decisions" and "Hard Rules"
as authoritative and do NOT re-litigate the reversed decisions (no
classification task, no full fine-tuning, demo only ONE simple metric).

Current locked direction: a Telco/Media/Gaming (TMG) grounded **research agent**
that answers questions using **Bing grounding** (WebIQ is in preview, Bing is the
fallback). We seed a golden set from a Hugging Face Q&A dataset (primary candidate
PatronusAI/financebench), have a frontier model (Opus / GPT-5.5) produce
gold-standard answers, then **distill** a cheaper model to match it, measured by a
single "Answer Quality Score," and continuously retrain as the data drifts.

My next task is: <PICK ONE — e.g.
  - "confirm the FinanceBench dataset card (columns/license/splits) and lock the dataset", or
  - "design the exact Answer Quality Score + promotion-gate thresholds", or
  - "scaffold the Foundry distillation loop (plan A) with a GPU LoRA/QLoRA fallback (plan B)", or
  - "write the Python to load the HF dataset, split it, and run a frontier-vs-cheap-model baseline eval">.

Work in this repo. Keep it simple and legible for an infra/data audience. Ask me
clarifying questions before writing significant code, and update CONTEXT.md and
the docs as decisions get made.
```

---

## 11. Glossary (for non-LLM-native teammates)

- **Frontier model:** the big, expensive, top-tier model (Claude Opus, GPT-5.5).
- **Distillation:** teach a small model to copy a big model's answers.
- **PEFT / LoRA / QLoRA:** cheap ways to fine-tune that only adjust a small set
  of extra weights (not the whole model).
- **Golden dataset:** the trusted "answer key" we grade models against.
- **Trace:** a logged record of a real model interaction (question + answer +
  context). We don't need real production traces for the demo — the HF dataset
  stands in.
- **Grounding / Bing grounding:** the model searches the live web and bases its
  answer on what it finds.
- **Drift:** the world changes, so yesterday's good model gets worse over time.
- **Promotion gate:** the pass/fail rules a new model must clear before it
  replaces the current one.
```
