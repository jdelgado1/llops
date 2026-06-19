# Handoff: Option A — Distilled Tool-Calling Agent (smaller beats larger)

> **Foundry-native, no GPU.** Training = Foundry **managed SFT** (upload a JSONL);
> serving = **serverless** deployment; eval = **BFCL AST** against the Foundry
> OpenAI-compatible endpoints.

## 0. TL;DR for the next agent
Build and measure a pipeline proving a **small distilled model matches/beats a frontier model at tool calling**, scored by an **objective metric (BFCL AST accuracy)**, wrapped in the Foundry/Fabric LLMOps lifecycle. **Do not** reintroduce the grounded-QA/RetrievalQA path or any "distillation lifts a fuzzy score" goal — those were explored and rejected (see §7).

## 1. Why this design (context)
- Demo thesis (Nikhil): *"model choice is a lifecycle decision."* Start with a frontier model, collect traces, build a golden set, distill a smaller model, continuously retrain on drift, promote via gates.
- We need the demo to show a **smaller model matching/beating a larger one at a task**, measured by a metric that **discriminates** models and is **objective** (no LLM-judge bias).
- Prior experiments: a grounded-QA task **saturated** (frontier = base = distilled ≈ 90%); closed-book showed a gap but distillation added **0 lift by design**. Conclusion: only **tool calling with AST accuracy** reliably yields the "smaller beats larger" headline — and it's already proven in prior work (a small Qwen3 beat GPT‑5.4 on BFCL-Python).
- Web search is honored by including a **`web_search` tool** in the toolset (Option A), not by switching tasks.

## 2. What "Option A" is (definition)
A **TMG ops/support agent** whose job is **tool/function calling**: given a user request + tool schemas (including a `web_search` tool and internal TMG APIs), emit the **correct function call (name + arguments)**. We **distill a frontier teacher's correct tool-call completions** into a small student via **Foundry managed SFT** (JSONL upload, no GPU), and benchmark all three models on BFCL-Python AST accuracy.

## 3. Locked configuration
| Piece | Value |
| --- | --- |
| Task | Tool/function calling (emit correct call name + args) |
| Scenario | TMG ops/support agent; tools = internal APIs + `web_search` |
| Eval / golden set | **BFCL v4, Python subset** via `bfcl-eval` |
| Metric | **BFCL AST accuracy** (right function + correct args) — single objective number |
| Teacher (large) | GPT‑5.4 (or Claude Opus) via Microsoft Foundry |
| Student (small) | **Qwen3-32B** — the only Foundry-fine-tunable Qwen size. ⚠️ **base inference unavailable** (fine-tune-only), so the "before" baseline is an optional proxy |
| Customization | **Foundry managed SFT distillation** (JSONL upload, serverless deploy, **no GPU**) — never full FT |
| Training prompts | ToolACE (data.json) — supplies requests + tool schemas |
| Training targets | **Frontier teacher's tool-call completions, rejection-sampled to AST-correct** (honors "frontier completions from prod traces") |
| Secondary metrics | cost ($/1k), p50/p95 latency, tokens/req (the "at lower cost" number) |

## 4. The training data (critical — honors Nikhil's bullet #3)
Do **not** SFT on raw ToolACE answers. Build distillation data as:
1. Take ToolACE prompts + tool schemas as the **prompt pool**.
2. Run the **frontier teacher (GPT‑5.4)** on them (this is the "production trace" generation step; in the live demo these come from Foundry Tracing of the hosted agent).
3. **Rejection-sample**: keep only completions whose tool call is **AST-correct** against the expected schema/answer.
4. Format kept pairs as SFT examples → this is the **SFT training set** (uploaded to the Foundry Fine-tune wizard).

This makes the data "the frontier model's completions from prod traces," and filtering to correct is what lets the student **match/exceed** the teacher.

## 5. Execution plan (step by step)
> Foundry-native — **no GPU**. Fine-tuning = Foundry managed SFT (upload JSONL,
> **Supervised** + **Global**); serving = **serverless** deployment; eval calls the
> Foundry OpenAI-compatible endpoints. Python runs in this repo's `.venv`.

1. **Establish baselines.**
   - Deploy the teacher **GPT-5.4** on Foundry. Run **AST eval** on the teacher; record accuracy.
   - ⚠️ **Base Qwen3-32B inference is NOT available** on Foundry (fine-tune-only), so you **cannot** measure an untuned base directly. The "before" column is **optional**: either omit it (headline = teacher vs distilled) or supply a **near-base proxy** — an off-task / format-primer fine-tune (knows the tool-call format, not our task).
   - ⚠️ Confirm each *deployed* model exposes the **function-calling API** (`tools` param + emits `tool_calls`) via `smoke_toolcalling.py`. If qwen needs special tool-call handling, the eval client already has a text-parse fallback.
2. **Generate distillation data.**
   - ToolACE prompts + tool schemas → call **GPT-5.4** → capture `tool_calls` → **AST-validate** against the expected answer → keep only correct → write **OpenAI-format SFT JSONL** (messages + tools + assistant tool-call target).
   - Include a `web_search` tool in the schemas so the agent learns when/how to invoke it.
3. **Distill (Foundry managed SFT).**
   - Upload the SFT JSONL to the Foundry **Fine-tune** wizard → **Supervised**, **Global**, base **Qwen3-32B** → train. (Flow already validated for qwen3-32b.)
4. **Deploy + eval the distilled student.**
   - Deploy the fine-tuned model (serverless), run **BFCL-Python AST eval** against its endpoint. Record AST accuracy.
5. **Comparison.**
   - Table: **frontier vs distilled Qwen3-32B** (plus the optional near-base proxy) on AST accuracy. Target: `distilled ≥ frontier` (and `≫` the proxy if present).
6. **Cost/latency.**
   - Measure $/1k, p50/p95 latency, tokens/req for the distilled qwen vs GPT-5.4 (watch over-thinking / excess reasoning tokens). This is the "parity at lower cost" number.
7. **Lifecycle wiring (loop).**
   - Foundry Hosted Agent runs the toolset; **Foundry Tracing** logs turns → **push production traces to Microsoft Fabric** (golden/drift set of record) → distill → eval vs golden+drift → **promotion gate** → checkpoints to Blob, eval rows to Azure SQL DB; Eventhouse drift signals trigger retrain. (Blu owns DB + dashboard.)
8. **Drift demo.**
   - Inject new tools / changed schemas → AST accuracy drops → retrain on fresh traced data → accuracy recovers. (Dashboard line moves.)

## 6. Success criteria
- **Headline:** distilled Qwen3-32B AST accuracy **≥ GPT‑5.4** on BFCL-Python (matches or beats).
- **Cost:** distilled-32B materially cheaper + faster than GPT‑5.4 at comparable accuracy.
- **Promotion gate:** ship distilled only if `AST(distilled) ≥ AST(frontier) − ε` AND cheaper AND faster AND safety pass.
- **Loop:** drift → retrain → recovery visibly demonstrated.

## 7. Guardrails / do-NOT
- **No full fine-tuning.** **Foundry managed SFT** distillation only (Supervised / Global).
- **One objective metric** for the headline: BFCL AST accuracy.
- **Don't** reintroduce: classification task, RetrievalQA/FinanceBench grounded-QA as the headline, or "distillation must lift a fuzzy score."
- Keep **train (ToolACE) and eval (BFCL) separate** — no leakage.
- Confirm each model's deployment exposes the **function-calling API** (`tools` + `tool_calls`) before scoring.

## 8. Pipeline assets (this repo) — BUILT
The tool-calling pipeline is built as new modules; the grounded-QA pipeline is
kept unmodified as the "why Option A" evidence.
- **`src/llmops/tooldata.py`** — tool-calling loader + deterministic by-id train/eval split (sample / hf / path).
- **`src/llmops/ast_check.py`** — the objective metric: BFCL-style AST accuracy (in-repo, no harness).
- **`src/llmops/tool_models.py`** — call a deployment with tools; parse `tool_calls` (GPT + qwen, text fallback).
- **`src/llmops/tool_traces.py`** — teacher → tool calls → AST-validate → keep correct (rejection sampling).
- **`src/llmops/tool_sft.py`** — correct traces → function-calling SFT JSONL (messages + tools + tool-call target) + validator.
- **`src/llmops/tool_eval.py`** — 3-way AST table (frontier / base / distilled) + cost/latency.
- **`src/llmops/fabric.py`** — push accepted traces into a Fabric Lakehouse (OneLake) with a local fallback.
- **`data/toolcalling_sample.jsonl`** — bundled offline TMG tool set so everything runs without network.

## 9. Open decisions (pick before running)
- `ε` for the promotion gate (set after baseline numbers exist).
- Whether to also run the optional FRAMES multi-hop "stretch" exhibit (default: **skip** for 6/26).

---
