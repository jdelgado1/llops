# Project Context & Handoff

> **Read this first.** This file is the single source of truth for the TMG
> Tool-Calling Agent LLMOps demo. It captures *what we're building, why, every
> decision we made (and reversed), and how to continue.* If you're an AI agent
> or a teammate picking this up cold, read this top-to-bottom before touching
> anything else.

Last updated: 2026-06-19 · Owner: **Jose Delgado**

---

## 1. The One-Paragraph Summary

We're building a **demo of the full LLM lifecycle on Azure / Microsoft Foundry**
for an internal session on **6/26**. The story: **"model choice is a lifecycle
decision."** Start with an expensive **frontier model** (GPT-5.4), use it to
produce gold-standard **tool calls**, then **distill a cheaper, smaller model**
(Qwen3-32B) that **matches or beats** it at the task — measured by **one
objective number (BFCL-style AST accuracy)** — and **continuously retrain** it as
tools/usage drift. The task is a **Telco/Media/Gaming (TMG) operations & support
agent** whose job is **tool/function calling**: given a request + tool schemas
(including a `web_search` tool), emit the **correct function call (name +
arguments)**. Training = **Foundry managed SFT** (upload a JSONL, **no GPU**);
serving = **serverless**; production traces flow **Foundry Tracing → Microsoft
Fabric** as the golden/drift set of record.

---

## 2. Who's Who

- **Jose (you):** owns the scenario, the objective metric, the trace→SFT→eval
  pipeline, and the fine-tune/distillation loop. (This repo.)
- **Nikhil Gopal:** lead / presenter. Set the direction ("model choice is a
  lifecycle decision"). Wants **simple, legible** concepts for a mixed
  infra/data audience.
- **Blu Gotlieb:** owns the database, Fabric/Eventhouse, and dashboards (drift &
  performance over time, likely Power BI).
- **Anthony Nevico:** stakeholder; gives feedback before the session.

---

## 3. What We're Actually Doing (Plain English)

1. Take a **tool-calling dataset** (a user request + the available tool schemas
   + the correct call). We ship a bundled TMG ops/support set; ToolACE/BFCL can
   be swapped in at scale.
2. Split it: a **training pile** and a hidden **held-out test pile** (by id, so
   no leakage).
3. Have the **frontier model (GPT-5.4)** answer the training prompts by calling
   tools → keep **only the calls that are objectively correct** (rejection
   sampling). These become the gold training traces.
4. **Score the frontier model** on the held-out pile → that's the quality bar
   (one number: **AST accuracy**, e.g. 88%).
5. **Score the base small model** → it does worse (e.g. 55%).
6. **Distill** the small model on the correct traces (Foundry managed SFT) → it
   **matches or beats** the frontier model (e.g. 90%) but far cheaper + faster.
   **That's the win.**
7. **Push production traces** from Foundry Tracing into **Microsoft Fabric**,
   schedule retraining, and store scores so we can chart quality over time.
8. **Drift story:** tools/schemas/usage change → AST accuracy drops →
   auto-retrain on fresh correct traces brings it back up.

> The metric is the scoreboard: **one AST accuracy percentage** that goes up as
> the model improves. It's **objective** (no LLM-judge bias), so a small model
> beating a large one is unambiguous. Keep it that simple.

---

## 4. Key Decisions (and Why)

| # | Decision | Why | Status |
| --- | --- | --- | --- |
| D1 | **New repo**, separate from the old Nebius function-calling repo | A clean Azure/Foundry repo tells a clean story | ✅ Locked |
| D2 | **Distillation (SFT)** as the customization method | The scenario produces frontier "teacher" tool calls; Foundry has native managed SFT; less labeling than DPO/RFT | ✅ Primary |
| D3 | **No full fine-tuning, ever** | Carried constraint. **Foundry managed SFT** only (no GPU) | ✅ Hard rule |
| D4 | Task = **tool/function calling** for a TMG ops/support agent | The only task that reliably shows **smaller beats larger** on an **objective** metric (already proven in prior work: a Qwen3 14B beat GPT-5.4 on BFCL-Python). Web search is honored via a `web_search` tool in the toolset, not by switching the task. **Reverses old D4/D5 (grounded research) — see reversed decisions.** | ✅ Locked |
| D5 | Metric = **BFCL-style AST accuracy** (right function + correct args) | One **objective** number — no LLM-judge bias, discriminates models cleanly. Implemented in-repo (`ast_check.py`) | ✅ Locked |
| D6 | Training = **Foundry managed SFT** on **Qwen3-32B**, serverless deploy, **no GPU** | The sub has **zero GPU quota**; fine-tuned qwen deploys **serverless** (proven). Qwen3-32B is the **only Foundry-fine-tunable** Qwen size. Note: its **base inference is unavailable** (fine-tune-only), so the "before" baseline is an optional proxy (see open questions) | ✅ Locked |
| D7 | **Train on rejection-sampled teacher traces; evaluate on a held-out, disjoint pool** | Keeps the headline honest (no leakage). Filtering teacher calls to AST-correct is what lets the student match/exceed the teacher | ✅ Locked |
| D8 | Data = bundled **TMG tool-calling sample** (offline, default) + **BFCL/ToolACE** at scale (`TOOLCALLING_SOURCE`) | Runs anywhere now; scales later. Train prompts (ToolACE-style) and eval (BFCL-style) stay separable | ✅ Locked |
| D9 | **Production traces: Foundry Tracing → Microsoft Fabric** (golden/drift of record) | Fabric is the data plane of record (Blu's domain); closes the lifecycle loop. Tenant-portable via env | ✅ Locked |

### Decisions we REVERSED (don't re-litigate)

- ❌ **Grounded research-QA as the task** (RetrievalQA / FreshQA / web-bench).
  Was the locked direction through 2026-06-17. **Reversed 2026-06-19** after the
  pipeline was built and run: grounded QA **saturated** — frontier = base =
  distilled ≈ **90%** in frozen-context, and distillation added **0 correctness
  lift by design** (it teaches grounded synthesis, not facts). It could not show
  "smaller beats larger." Replaced by **tool calling** (D4), which does. The
  grounded-QA code is **kept in this repo as the evidence** for *why* we pivoted
  (see §6 Legacy).
- ❌ **LLM-judge "Answer Quality Score" as the headline.** It was subject to
  judge bias (judge = teacher grading its own student). Replaced by **objective
  AST accuracy** (D5).
- ❌ **GPU Managed Compute + LoRA/QLoRA** as the training path. The subscription
  has **zero GPU quota**. Replaced by **Foundry managed SFT** (D6). No
  Axolotl / SGLang / GPU anywhere.
- ❌ **Classification task** (Banking77, etc.). Long-reversed; do not reintroduce.

---

## 5. The Source-of-Truth Context (from Nikhil)

The original session agenda and Nikhil's verbal guidance live in
[`context/teams-chat.md`](context/teams-chat.md) (**never edit that file**). The
thesis and lifecycle are unchanged; only the **task** changed from grounded QA to
tool calling, and the **metric** from a fuzzy quality score to objective AST
accuracy — both to make "smaller beats larger" demonstrable.

### Target architecture (Nikhil's, still current)

| Layer | Service |
| --- | --- |
| AI provider | Microsoft Foundry |
| Agent runtime | Foundry Hosted Agents (toolset incl. `web_search` / Web IQ) |
| Evals & tracing | Foundry Tracing + the in-repo AST eval |
| Model customization | **Foundry managed SFT** on Qwen3-32B (serverless, no GPU) |
| Golden / drift dataset | **Microsoft Fabric** (Foundry Tracing → OneLake) |
| Checkpoint storage | Azure Blob Storage |
| Eval results & completions | Azure SQL DB |
| Failures / latency / drift signals | Eventhouse |
| Dashboards | Power BI |

---

## 6. Current State of This Repo

```
tmg-research-agent-llmops/
├── README.md                         # Overview, thesis, stack, the one objective metric
├── CONTEXT.md                        # ← this file
├── .env.example                      # config (3 model slots + Fabric + data source)
├── requirements.txt
├── data/
│   └── toolcalling_sample.jsonl      # bundled offline TMG tool-calling set (BFCL-style)
├── docs/
│   ├── scenario.md                   # the TMG ops/support tool-calling scenario
│   ├── golden-dataset-and-eval.md    # data + AST metric + train/eval split
│   ├── distillation-loop.md          # the Foundry-native loop + Azure/Fabric architecture
│   ├── handoff-option-a.md           # the Option A spec (the "why")
│   └── build-plan.md                 # actionable build plan + tenant portability + Fabric push
├── src/llmops/
│   ├── config.py                     # env-driven Settings (3 models, Fabric, data source)
│   ├── ast_check.py                  # ★ the objective metric (BFCL-style AST accuracy)
│   ├── tooldata.py                   # ★ tool-calling loader + by-id train/eval split
│   ├── tool_models.py                # ★ call a deployment with tools, parse calls (GPT + qwen)
│   ├── tool_traces.py                # ★ teacher → tool calls → AST-validate → keep correct
│   ├── tool_sft.py                   # ★ traces → function-calling SFT JSONL (+ validator)
│   ├── tool_eval.py                  # ★ 3-way AST eval + cost/latency
│   ├── fabric.py                     # ★ push accepted traces to Microsoft Fabric (OneLake)
│   ├── models.py                     # shared OpenAI-client helper
│   └── (legacy) data.py, teacher.py, traces.py, sft_dataset.py, evaluate.py,
│       distill_eval.py, judge.py     # grounded-QA pipeline — kept as the "why we pivoted" evidence
└── scripts/
    ├── smoke_toolcalling.py          # prove tool calling + AST end-to-end
    ├── gen_tool_traces.py            # generate rejection-sampled teacher traces
    ├── build_tool_sft.py             # build the function-calling SFT dataset
    ├── run_tool_eval.py              # run the 3-way AST eval
    ├── push_to_fabric.py             # push accepted traces into Fabric
    └── (legacy) smoke_grounding.py, gen_traces.py, build_sft.py,
        run_baseline.py, run_distill_eval.py
```

**★ = the Option A (tool-calling) pipeline — the primary path.** The legacy
grounded-QA modules are retained, unmodified, as the empirical evidence behind
the pivot (saturation + 0 distillation lift).

---

## 7. Jose's Assigned Tasks

1. ✅ Define the **scenario** — `docs/scenario.md` (TMG ops/support tool agent).
2. ✅ Define the **objective metric + data** — `docs/golden-dataset-and-eval.md`
   (AST accuracy; bundled tool-calling set + BFCL/ToolACE at scale).
3. ✅ Build the **trace → SFT → eval** pipeline (this repo, `src/llmops/*` ★).
4. ⬜ Run it on Foundry: baselines → distill (managed SFT) → 3-way AST table →
   cost/latency.
5. ⬜ Wire **Foundry Tracing → Fabric** + scheduled retraining + drift demo
   (with Blu).

---

## 8. Open Questions to Resolve Next

- [ ] Confirm the *deployed* qwen (fine-tuned or proxy) exposes the
      **function-calling API** (`tools` param → emit `tool_calls`). Base Qwen3-32B
      inference is **unavailable**, so verify on a fine-tuned/proxy deployment
      with `smoke_toolcalling.py`.
- [ ] `ε` for the promotion gate (set after baseline AST numbers exist).
- [ ] **Baseline ("before") strategy:** base Qwen3-32B can't be served, so either
      (a) omit the baseline column (headline = teacher vs distilled), or (b) use a
      **format-primer** fine-tune (generic, off-distribution tool calls) as a
      near-base proxy. Lead with (b) if we want the before→after drama.
- [ ] Scale data source: bundled sample (now) → ToolACE/BFCL via
      `TOOLCALLING_SOURCE` for a real SFT run.
- [ ] Align with **Blu** on the Fabric table schema + Eventhouse drift signals.

---

## 9. Hard Rules / Guardrails

- **No full fine-tuning.** **Foundry managed SFT** distillation only (no GPU).
- **Task = tool/function calling**, scored by **one objective metric (AST
  accuracy)**. Don't reintroduce grounded-QA-as-the-task, classification, or a
  fuzzy LLM-judge headline.
- **Train on rejection-sampled teacher traces; evaluate on a disjoint held-out
  pool.** Never train on eval items (split is deterministic by id).
- **Keep quality (AST %) and performance (latency/tokens) as separate numbers.**
- **Production traces of record live in Fabric** (Foundry Tracing → OneLake).
- Concepts > implementation details (Nikhil's words). Don't over-engineer.

---

## 10. Prompt to Continue This Work

```text
You are helping me (Jose) build an Azure / Microsoft Foundry demo of the full
LLM lifecycle for an internal session on 6/26. Before doing anything, read
CONTEXT.md end-to-end — it has the full decision history, the locked direction,
and open questions. Treat "Key Decisions" and "Hard Rules" as authoritative and
do NOT re-litigate reversed decisions (no grounded-QA-as-the-task, no
classification, no GPU/LoRA, no LLM-judge headline, no full fine-tuning).

Locked direction (Option A): a Telco/Media/Gaming (TMG) ops/support agent whose
task is TOOL/FUNCTION CALLING. A frontier teacher (GPT-5.4) produces tool calls;
we rejection-sample to AST-correct ones and DISTILL a small student (Qwen3-32B)
via Foundry MANAGED SFT (JSONL upload, no GPU, serverless deploy). The single
headline metric is BFCL-style AST accuracy (objective). We prove distilled ≥
frontier ≫ base, at lower cost/latency. Production traces flow Foundry Tracing →
Microsoft Fabric. The pipeline lives in src/llmops/*: ast_check, tooldata,
tool_models, tool_traces, tool_sft, tool_eval, fabric.

My next task is: <PICK ONE — e.g.
  - "run smoke_toolcalling against the base qwen to confirm it emits tool_calls", or
  - "generate teacher traces at scale from ToolACE and build the SFT JSONL", or
  - "run the 3-way AST eval after I deploy the distilled qwen", or
  - "wire the Foundry Tracing → Fabric push and the drift/retrain demo">.

Work in this repo. Keep it simple and legible for an infra/data audience. Update
CONTEXT.md and the docs as decisions get made.
```

---

## 11. Glossary (for non-LLM-native teammates)

- **Frontier model:** the big, expensive, top-tier model (GPT-5.4 here).
- **Tool / function calling:** instead of writing prose, the model emits a
  structured **call** — a function name + JSON arguments — that an app executes
  (e.g. `create_support_ticket(customer_id="CU-9012", priority="high")`).
- **AST accuracy:** the objective metric — is the predicted call the *right
  function with the right arguments*? (BFCL methodology.) One number, no judge.
- **Rejection sampling:** keep only the teacher's calls that are objectively
  correct; train on those. This is what lets the student match/beat the teacher.
- **Distillation / SFT:** teach a small model to copy a big model's (correct)
  calls. We use **Foundry managed SFT** — upload a JSONL, no GPU.
- **Serverless deploy:** a fine-tuned model served pay-per-token with no GPU
  capacity to reserve.
- **Golden / drift dataset:** the trusted set of correct traces (in Fabric) we
  retrain on and grade against; "drift" = the world/tools change, so the score
  drops until we retrain.
- **Promotion gate:** the pass/fail rules a new model must clear (AST ≥ frontier
  − ε, cheaper, faster, safety pass) before it replaces the current one.
