# Build Plan — Option A (Distilled Tool-Calling Agent), Foundry-native

> Companion to [`handoff-option-a.md`](handoff-option-a.md) (the *why* + spec).
> This is the **actionable checklist**: what to set up in a (new) tenant, what
> changes to make to the existing pipeline, and the end-to-end run order.
> **No GPU** — training is Foundry managed SFT (JSONL); serving is serverless.

---

## A. Tenant portability — set this up in any tenant first
The repo code is tenant-agnostic: it reads the endpoint + deployment names from
`.env`. Porting to a new tenant = provision the resources below, then fill `.env`.

### A1. Azure / Foundry resources
| Resource | Notes |
| --- | --- |
| Resource group | e.g. `rg-llmops` |
| **Foundry resource + project** (AI Services) | gives the project endpoint `https://<res>.services.ai.azure.com/api/projects/<proj>` |
| **Teacher deployment** | `gpt-5.4` (Global Standard). ⚠️ GPT-5.x may need a **quota tier** request on some subs (Tier 5/6 have it by default). |
| **Base student for fine-tuning** | `Qwen3-32B` (Direct-from-Azure, **fine-tune-only — base inference unavailable**). Region must support **Global** SFT (e.g. East US 2). |
| **Fine-tuned student deployment** | created after SFT → **serverless (Standard/Global)**, no GPU quota |
| **Web Search tool** | enable subscription preview feature if used (no extra resource) |

### A2. Permissions (you, in the new tenant)
- **Owner on the resource group** (create resources + role assignments), or
  **Subscription Owner/Contributor** for sub-level prereqs (provider/feature registration, quota).
- **Azure AI Developer / Foundry Owner** on the *project* to deploy models.
- **Fabric:** a **Fabric workspace + capacity (F SKU)** and a **workspace role**
  (Admin/Member) — Fabric is a *separate permission plane* from Azure RBAC + needs a license.

### A3. `.env` (copy from `.env.example`, fill per tenant)
```
FOUNDRY_PROJECT_ENDPOINT=https://<res>.services.ai.azure.com/api/projects/<proj>
TEACHER_MODEL=gpt-5.4
STUDENT_BASE_MODEL=qwen3-32b
STUDENT_FINETUNED_DEPLOYMENT=        # set after you deploy the distilled model
FABRIC_WORKSPACE_ID=                 # for the trace push
FABRIC_LAKEHOUSE=                    # table/lakehouse name
```

### A4. Portability smoke test (run after setup)
- `az login` (new tenant) → confirm subscription.
- Call the teacher + base/fine-tuned deployments once (chat completions) to confirm auth + endpoints.
- Confirm each deployment exposes the **function-calling API** (`tools` param → emits `tool_calls`).

---

## B. Pipeline changes (grounded-QA → tool-calling) — BUILT
The tool-calling pipeline is built as **new modules** (the grounded-QA code is
**kept as the "why Option A" evidence**). Status below:

| Concern | Module | Status |
| --- | --- | --- |
| Tool-calling loader + by-id split | `src/llmops/tooldata.py` | ✅ built (sample / hf / path) |
| Objective metric (BFCL-style AST) | `src/llmops/ast_check.py` | ✅ built (in-repo, no harness) |
| Call a deployment with tools (GPT+qwen) | `src/llmops/tool_models.py` | ✅ built (+ text-parse fallback) |
| Teacher traces, AST-validated/kept | `src/llmops/tool_traces.py` | ✅ built (rejection sampling) |
| Function-calling SFT JSONL (+validator) | `src/llmops/tool_sft.py` | ✅ built |
| 3-way AST eval + cost/latency | `src/llmops/tool_eval.py` | ✅ built |
| Push accepted traces to Fabric | `src/llmops/fabric.py` | ✅ built (OneLake + local fallback) |
| Bundled offline TMG tool set | `data/toolcalling_sample.jsonl` | ✅ built |

Keep: `config.py`, `models.py` (client). Validated offline end-to-end
(loader → AST → SFT → Fabric-prep). Remaining = run on Foundry (§C).

---

## C. End-to-end run order
1. **Baselines** — AST eval on **GPT-5.4**. Base Qwen3-32B inference is
   unavailable, so the "before" column is **optional**: omit it, or use a
   **format-primer** near-base proxy. Record AST accuracy.
2. **Generate distillation data** — ToolACE prompts → GPT-5.4 `tool_calls` → AST-validate → keep correct → **SFT JSONL**.
3. **Distill** — upload JSONL to Foundry **Fine-tune** wizard: **Supervised / Global / Qwen3-32B** → train.
4. **Deploy** the fine-tuned model (serverless) → set `STUDENT_FINETUNED_DEPLOYMENT`.
5. **Eval distilled** — BFCL-Python AST accuracy against its endpoint.
6. **Three-way table** — frontier vs base vs distilled. Target: `distilled ≥ frontier ≫ base`.
7. **Cost/latency** — $/1k, p50/p95, tokens/req for distilled vs GPT-5.4.
8. **Promotion gate** — ship distilled only if `AST(distilled) ≥ AST(frontier) − ε` AND cheaper AND faster AND safety pass.
9. **Drift demo** — change/add tool schemas → AST drops → retrain on fresh traces → recovers.

---

## D. Foundry → Fabric: push production-grade traces (required step)
In production the training/golden data comes from **Foundry Tracing** of the hosted
agent, landed in **Microsoft Fabric** as the golden/drift set of record (Blu owns
the DB + Power BI side).

**Flow:**
1. Hosted agent (toolset incl. `web_search`) serves requests; **Foundry Tracing** captures each turn (prompt, tools offered, `tool_calls`, outcome).
2. **AST-validate** each traced tool call; tag accepted (correct) vs rejected.
3. **Push accepted traces to a Fabric table** (Lakehouse/Warehouse) — the golden/drift dataset of record.
   - Options: Foundry Tracing → App Insights/Event Hub → **Fabric Eventstream/Eventhouse**, or a small exporter that writes traces to **OneLake/Lakehouse** via the Fabric API (`FABRIC_WORKSPACE_ID` + `FABRIC_LAKEHOUSE`).
4. Distillation reads the **accepted** Fabric rows → SFT JSONL → retrain (closes the loop).
5. Eval rows → **Azure SQL DB**; drift signals → **Eventhouse** → trigger retrain; checkpoints → **Blob**.

> Coordinate the Fabric/DB/Eventhouse/dashboard wiring with **Blu** (his domain).
> Don't block the distillation loop on Fabric — generate JSONL locally first, add
> the Fabric sink in parallel.

---

## E. Open decisions
- `ε` for the promotion gate (set after baselines exist).
- Student = **Qwen3-32B** (the only Foundry-fine-tunable Qwen). Baseline: omit, or build a **format-primer** near-base proxy — decide after seeing teacher/distilled numbers.
- Optional FRAMES multi-hop "stretch" exhibit — default **skip** for 6/26.
