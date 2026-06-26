# Golden Dataset & Evaluation — Tool Calling + AST Accuracy

> Task #2. The data, the train/eval split, and the **one objective metric**.

## The one metric: AST accuracy

The whole demo tracks a single number — **AST accuracy** — using the **BFCL**
(Berkeley Function-Calling Leaderboard) methodology, implemented in-repo in
[`ast_check.py`](../src/llmops/ast_check.py):

> A predicted tool call is **correct** iff it
> 1. names the **right function**,
> 2. includes every **required parameter**,
> 3. gives each parameter a value among the reference's **acceptable values**
>    (BFCL ground-truth allows several valid values per argument), and
> 4. **invents no parameter** the schema doesn't define.
>
> For **parallel / multiple-call** items, the prediction is a *list* of calls and
> is correct iff it matches the reference list as a set (same count, each
> reference call matched by a distinct predicted call).
>
> **AST accuracy** for a model = % of held-out items whose call(s) are correct.

Why this metric:

- **Objective** — no LLM judge, so no bias when the teacher grades a student.
- **Discriminates** — models actually differ here (unlike grounded QA, which
  saturated), so "smaller beats larger" is visible and unambiguous.
- **Legible** — one percentage that goes up as the model improves.

Performance (p50/p95 latency, output tokens) is tracked **separately** by
[`tool_eval.py`](../src/llmops/tool_eval.py) — never folded into the quality
number.

## The data

Each item (BFCL-style) is:

```json
{
  "tid": "tmg_multiple_0",
  "category": "simple | multiple | parallel | parallel_multiple",
  "messages": [{"role": "user", "content": "<request>"}],
  "tools":    [ <OpenAI tool schemas offered to the model> ],
  "reference":[ {"name": "<fn>", "arguments": {"<param>": ["<acceptable>", ...]}} ]
}
```

Sources (`TOOLCALLING_SOURCE` / `--source`):

| Source | Role |
| --- | --- |
| `sample` (default) | Bundled offline TMG ops/support set — runs anywhere, no network |
| a path | A directory/JSONL of the same shape (e.g. exported BFCL / ToolACE) |
| `hf` | Pull the Berkeley Function-Calling Leaderboard at scale (best-effort) |

The bundled set covers `simple`, `multiple`, and `parallel` categories plus a
`web_search` item, so the metric and pipeline exercise every code path offline.
For a real SFT run, use **ToolACE** for training prompts/schemas and **BFCL** for
eval, keeping train and eval separable.

## The train/eval split (no leakage)

[`tooldata.py`](../src/llmops/tooldata.py) splits the source **by item id** (a
stable hash) into:

- a **trace-generation pool** → the teacher answers these → (rejection-sampled)
  becomes training data, and
- a **held-out eval pool** → only ever used to score AST accuracy / gate.

The split is deterministic, so it's reproducible run-to-run and the held-out pool
is **never** trained on.

## How the golden training set is built

We do **not** SFT on raw reference answers. We build distillation data by
**rejection-sampling the frontier teacher** ([`tool_traces.py`](../src/llmops/tool_traces.py)):

1. Take the trace-pool prompts + tool schemas.
2. Run the **frontier teacher (GPT-5.4)** with `tool_choice="required"`.
3. **AST-validate** each teacher call against the reference; **keep only correct
   ones**.
4. Format kept calls as **function-calling SFT JSONL**
   ([`tool_sft.py`](../src/llmops/tool_sft.py)): `messages` + `tools` + an
   assistant target carrying `tool_calls`.

Filtering the teacher to **AST-correct** calls is precisely what lets the small
student **match or exceed** the teacher. In the live demo these traces come from
**Foundry Tracing** of the hosted agent (and are pushed to **Fabric** as the
golden/drift set of record).

## The headline comparison

[`tool_eval.py`](../src/llmops/tool_eval.py) scores three deployments on the
held-out pool:

| Model | Slot | Expectation |
| --- | --- | --- |
| `teacher` = GPT-5.4 | frontier quality bar | high AST |
| `baseline` = near-base Qwen3-32B proxy *(optional)* | the "before" | low AST |
| `distilled` = SFT'd Qwen3-32B | the "after" | **≥ teacher** |

> **No directly-measurable base:** Foundry serves Qwen3-32B only after
> fine-tuning ("base model inference is not currently available"). So `baseline`
> is **optional** — either omit it (headline = teacher vs distilled) or supply a
> **format-primer** fine-tune (generic, off-distribution tool calls; knows the
> *format* but not our *task*) as a near-base proxy. `tool_eval.py` runs with or
> without it.
Target headline: **`distilled ≥ teacher`** (and `≫ baseline` when present), at **lower latency/tokens**.

## Promotion gate

Ship the distilled model only if:

- `AST(distilled) ≥ AST(frontier) − ε` (set `ε` after baselines exist), **and**
- it's **cheaper** (tokens/$) **and faster** (p50/p95), **and**
- it passes a **safety** check.

## Honesty notes

- We **evaluate on a held-out pool the student never saw**, so the headline is
  not leakage.
- The student is trained on the **teacher's correct calls**, so it learns the
  *behavior* (right tool + args), which is the thing AST accuracy measures —
  legitimately, because train and eval items are disjoint.
