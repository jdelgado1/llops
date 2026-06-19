# TMG Tool-Calling Agent — LLMOps Lifecycle Demo

A reference implementation of the **full LLM lifecycle on Azure / Microsoft
Foundry**, framed around a **Telco, Media & Gaming (TMG) operations & support
agent** whose job is **tool / function calling**.

The thesis (per the 6/26 session agenda): **model choice is a lifecycle
decision** — quality, latency, cost, safety, drift, and retraining readiness.
We start with a frontier model, collect tool-call traces, build a golden set,
run continuous evaluation, and **distill** a smaller, cheaper model that we
promote only when it passes quality / cost / latency / safety gates.

> **Start here:** [`CONTEXT.md`](CONTEXT.md) is the full handoff — decision
> history, locked direction, open questions, and a ready-to-paste prompt to
> continue the work. Read it before touching anything. The Option A spec is in
> [`docs/handoff-option-a.md`](docs/handoff-option-a.md); the actionable build +
> tenant-portability plan is in [`docs/build-plan.md`](docs/build-plan.md).

## Scenario in one line

A **TMG operations & support agent**: given a user request (e.g. *"Open a high
priority ticket for customer CU-9012 about a billing error"*) and a set of tool
schemas (internal APIs + a `web_search` tool), the agent emits the **correct
function call** — the right tool with the right arguments. Tools and usage
patterns change over time, which is exactly what justifies the continuous-eval +
auto-retrain loop. Details in [`docs/scenario.md`](docs/scenario.md).

## What we measure (one objective number)

The whole demo tracks a single **AST accuracy** score (0–100%): *did the model
call the right function with the right arguments?* This is the **BFCL** (Berkeley
Function-Calling Leaderboard) methodology, implemented in-repo
([`src/llmops/ast_check.py`](src/llmops/ast_check.py)) so it's legible and
self-contained. It is **objective** — no LLM-judge bias — so "a small model beats
a large one" is unambiguous. (Latency and token cost are tracked **separately**,
never conflated with quality.)

## Why tool calling (and why we pivoted here)

We first built a **grounded research-QA** pipeline (kept in this repo as the
`legacy` modules). Run end-to-end on Foundry, it **saturated**: frontier, base,
and distilled models all scored ≈ 90% with frozen context, and distillation
added **0 correctness lift by design** (it teaches grounded *synthesis*, not
facts). It could not demonstrate "smaller beats larger."

**Tool calling does** — on an objective metric — and it's already proven (a
Qwen3-14B beat GPT-5.4 on BFCL-Python in prior work). So Option A makes
tool calling the task and **AST accuracy** the scoreboard. See the reversed
decisions in [`CONTEXT.md`](CONTEXT.md).

## The pipeline (train on traces, eval on a disjoint held-out pool)

The core rule: **we never train on eval items.** The tool-calling source is split
*by id* into a trace-generation pool and a held-out eval pool.

| Stage | Module | What it does |
| --- | --- | --- |
| Data | [`tooldata.py`](src/llmops/tooldata.py) | Load tool-calling items; deterministic train/eval split |
| Metric | [`ast_check.py`](src/llmops/ast_check.py) | Objective AST accuracy (right function + args) |
| Call | [`tool_models.py`](src/llmops/tool_models.py) | Call a deployment with tools; parse calls (GPT + qwen) |
| Traces | [`tool_traces.py`](src/llmops/tool_traces.py) | Teacher → tool calls → **AST-validate** → keep correct |
| SFT | [`tool_sft.py`](src/llmops/tool_sft.py) | Correct traces → function-calling SFT JSONL (+ validator) |
| Eval | [`tool_eval.py`](src/llmops/tool_eval.py) | 3-way AST table (teacher/base/distilled) + cost/latency |
| Fabric | [`fabric.py`](src/llmops/fabric.py) | Push accepted traces into Microsoft Fabric (OneLake) |

Rejection-sampling the teacher's calls to the **AST-correct** ones is what lets
the small student **match or beat** the teacher. Details in
[`docs/golden-dataset-and-eval.md`](docs/golden-dataset-and-eval.md).

## Quickstart

```powershell
pip install -r requirements.txt
az login
Copy-Item .env.example .env   # then fill FOUNDRY_PROJECT_ENDPOINT + deployment names

# 1. Prove tool calling + AST scoring work end-to-end (uses the bundled set)
python scripts/smoke_toolcalling.py

# 2. Generate rejection-sampled teacher traces
python scripts/gen_tool_traces.py --limit 50

# 3. Build the function-calling SFT dataset (upload this to Foundry Fine-tune)
python scripts/build_tool_sft.py --traces artifacts/tool-traces-XXNN.jsonl

# 4. After you deploy the distilled qwen, run the 3-way AST eval
python scripts/run_tool_eval.py --limit 20

# 5. Push accepted traces into Microsoft Fabric
python scripts/push_to_fabric.py --traces artifacts/tool-traces-XXNN.jsonl
```

The bundled [`data/toolcalling_sample.jsonl`](data/toolcalling_sample.jsonl)
runs everything offline-by-default. For a real SFT run, point
`TOOLCALLING_SOURCE` at BFCL/ToolACE (`hf`) or a local BFCL-format directory.

## Stack (target)

| Layer | Service |
| --- | --- |
| AI provider | Microsoft Foundry |
| Agent runtime | Foundry hosted agents + toolset (incl. `web_search` / Web IQ) |
| Evals & tracing | Foundry tracing + the in-repo AST eval |
| Model customization | **Foundry managed SFT** on **Qwen3-32B** (serverless, **no GPU**) |
| Golden / drift dataset | Microsoft Fabric (Foundry Tracing → OneLake) |
| Checkpoint storage | Azure Blob Storage |
| Eval results & completions | Azure SQL DB |
| Failure / latency / drift signals | Eventhouse |
| Dashboards | Power BI |

## Model customization — Foundry managed SFT (no GPU)

The subscription has **zero GPU quota**, and fine-tuned Qwen models deploy
**serverless** on Foundry — so the whole loop is GPU-free:

1. Upload the function-calling SFT JSONL to the Foundry **Fine-tune** wizard
   (**Supervised**, **Global**, base **Qwen3-32B** — the only Foundry-fine-tunable
   Qwen size).
2. Train, then **deploy serverless**; set `STUDENT_FINETUNED_DEPLOYMENT`.
3. Run the 3-way AST eval. Target: `distilled ≥ frontier`, cheaper + faster.

> **No base baseline:** Foundry says base Qwen3-32B *"is only supported for
> fine-tuning; base model inference is not currently available."* So the "before"
> column is **optional** — supply an off-task / format-primer fine-tune as a
> near-base proxy, or omit it and let the headline stand on **teacher vs
> distilled**.

This stays within the **SFT** guardrail (never full-parameter training) and is
the most "cohesive Azure" path. No Axolotl / SGLang / LoRA-on-GPU anywhere.

## Constraints carried from prior project

- **No full fine-tuning.** Customization is **distillation (SFT)** only.
- **One objective metric** for the headline: AST accuracy.
- Keep correctness (AST %) and serving performance (latency/cost) as
  **separate** measurements; don't conflate them.

## Status

Pipeline built and validated offline end-to-end (loader → AST → traces → SFT →
Fabric-prep). Next: run on Foundry (baselines → managed SFT → 3-way eval) and
wire the Foundry Tracing → Fabric push + drift/retrain demo with Blu.
