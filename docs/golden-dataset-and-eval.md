# Golden Dataset & Evaluation Metrics

> Task #2 from Nikhil's list: *Identify golden dataset and eval metric to
> benchmark frontier model vs smaller OSS model.*

## Design Principle (Two Layers)

The golden dataset has two layers that work together:

1. **Seed layer — an existing Hugging Face Q&A dataset.** This gives us
   ready-made **questions + reference answers** in a TMG-relevant domain, so we
   don't hand-author hundreds of questions and we get a **reproducible** answer
   key. See [Choosing the Hugging Face Seed Dataset](#choosing-the-hugging-face-seed-dataset).
2. **Trace layer — the frontier model's answers to those questions.** The
   frontier model (the **teacher**) answers the seed questions *with Bing
   grounding*; its high-quality, human-reviewed completions become the
   **evaluation reference** and (later) the **distillation training target**.
   This matches the agenda's "capture prompts, retrieval context, completions,
   citations, human ratings" step.

> **In plain terms:** one Hugging Face dataset gives us the questions and a
> trusted answer key; the frontier model shows us the "gold standard" answer
> style. We then teach a cheaper model to match it.

## Headline Metric (Keep It Simple)

Per Nikhil's guidance — the audience includes infra/data folks who aren't
LLM-native, so **simple is better** — the demo tracks **one headline number**:

> **Answer Quality Score** = a single 0–100% score for "is the answer correct
> and supported by what it retrieved?" (groundedness/correctness, judged against
> the reference answer).

That one number is what goes up on the dashboard over time. The richer metric
suite below still exists under the hood (and matches the formal agenda), but we
**demo one clean line**, optionally blending a few signals into a single
coefficient as Nikhil suggested.

## Choosing the Hugging Face Seed Dataset

**What kind of dataset to look for:** a **Question Answering (QA)** dataset whose
rows contain, at minimum, a **question** and a **reference answer** — and ideally
a **source document / evidence passage** and/or **citations** (so we can score
groundedness, not just string match).

**How to find them on Hugging Face:**

1. Go to <https://huggingface.co/datasets> and set the **Task** filter to
   *Question Answering*.
2. Add keywords relevant to TMG: `finance`, `earnings`, `news`, `telecom`,
   `customer support`, `RAG`, `grounded`, or `attributed QA`.
3. Open the **Dataset Viewer** and confirm the columns include something like
   `question` + `answer` (+ `context` / `evidence` / `citations`). Check the
   **license** allows our use.

**Recommended candidates (verified to exist on HF):**

| Dataset | Why it fits TMG | Structure |
| --- | --- | --- |
| **`PatronusAI/financebench`** *(primary pick)* | Analyst-style questions over **real company financial filings** (revenue, margins, earnings) — exactly the "market/competitive intelligence" research a TMG analyst does, and telco/media/gaming are public companies. Recognized benchmark → credible to the audience. | question + answer + supporting evidence (citations) |
| **`Maluuba/newsqa` / `lucadiliello/newsqa`** | **News** reading-comprehension QA — gives the media/news "drift" flavor. | question + answer + source news article |

**Recommendation:** use **FinanceBench** as the primary seed — it's QA over real
company financials (think telecom/media/gaming earnings), already has reference
answers + cited evidence, and is a known benchmark. Keep **NewsQA** as the
news-flavored alternative if we want a more obvious "the world changed" drift
story.

> FinanceBench is small (~150 curated questions). That's *fine* for a golden
> test set; for the **training** side of distillation we expand it with more of
> the same-style Q&A (frontier-generated) as needed.

## Record Schema

Each golden record captures the full grounded-research interaction:

| Field | Description |
| --- | --- |
| `query_id` | Unique id |
| `timestamp` | Trace time — **critical** for drift analysis |
| `persona` | Analyst / PM / GTM / strategy / exec |
| `segment` | telco / media / gaming |
| `query_type` | competitive-summary / regulatory-brief / sentiment-synthesis / metric-briefing |
| `question` | The user prompt |
| `retrieved_context` | Bing grounding results (+ any tool outputs), **snapshotted at trace time** |
| `frontier_completion` | Teacher answer (briefing text) |
| `citations` | Sources actually used (URLs + titles) |
| `human_rating` | 1–5 reviewer score |
| `pass_fail` | Reviewer accept/reject |
| `failure_labels` | hallucination / stale / missing-citation / off-topic / unsafe / format |
| `latency_ms` | End-to-end latency of the trace |
| `prompt_tokens`, `completion_tokens` | Token counts |
| `cost_usd` | Estimated request cost |

> **Why snapshot `retrieved_context`:** for a *live*-grounded agent, the
> "correct" answer depends on what was retrieved at that moment. We evaluate (and
> later distill) the model's **reasoning / synthesis / citation over provided
> context** — not its ability to re-retrieve. Freezing the context makes eval
> reproducible and the distillation target honest.

## Dataset Composition

- **Core golden set (~200–400 records):** human-rated, high-quality examples,
  balanced across the 3 segments and 4 query types. Reviewer-accepted only.
- **Drift slice (rolling):** the most recent N weeks of traces, held out to test
  **temporal robustness** and to surface quality decay.
- **Frozen regression set:** a stable subset never changed, for apples-to-apples
  model-vs-model comparison over time.
- **(Optional) adversarial/safety slice:** prompts probing hallucination,
  stale-data traps, and unsafe requests.

Stored in **Microsoft Fabric** (per the target architecture); eval runs and
completions land in **Azure SQL DB**.

## Evaluation Metrics

> **Reminder:** for the demo we surface the **single Answer Quality Score**
> above. The table below is the "full version" (matches the formal agenda) that
> the headline number is distilled from — show it only if the audience wants
> depth.

### Quality (full version — Foundry built-in evaluators)

| Metric | What it measures | Why it matters here |
| --- | --- | --- |
| **Groundedness** | Is the answer supported by `retrieved_context`? | Core anti-hallucination signal for a research agent |
| **Citation quality** | Precision/recall of citations vs sources actually used; correct attribution | Briefings are only trustworthy if sourced |
| **Relevance / task adherence** | Answers the question; follows house format | Usability for analysts |
| **Fluency / coherence** | Readability of the briefing | Exec-facing output |
| **Safety / content safety** | Harmful / policy-violating content | Promotion hard-gate |
| **Correctness vs golden (optional)** | Semantic similarity / LLM-as-judge vs `frontier_completion` | Direct teacher-parity check |

### Operational (secondary — but decisive for "model choice")

| Metric | Notes |
| --- | --- |
| **p50 / p95 latency, TTFT** | User-perceived responsiveness |
| **Cost per 1k queries** | The headline savings of a distilled model |
| **Throughput** | Concurrency behavior under load |

## Frontier vs Distilled — Comparison Protocol

Run **both** models over the **frozen regression set** + the **current drift
slice**, scoring every quality + operational metric. Present side-by-side.

The story we expect to tell:

- Distilled model **matches** frontier on groundedness / citation / relevance
  **on this narrow TMG task**, while …
- … winning decisively on **cost** and **latency**, and …
- … staying within safety gates.

## Promotion Gate (Decision Rule)

A distilled candidate is promoted to production **only if all hold**:

```
groundedness(candidate)      >= groundedness(frontier) - epsilon
citation_f1(candidate)       >= citation_threshold
task_adherence(candidate)    >= adherence_threshold
safety(candidate)            == PASS                 (hard gate)
cost_per_1k(candidate)       <  cost_per_1k(frontier) * cost_ratio
p95_latency(candidate)       <  p95_latency_budget
```

This rule **is** the demo's punchline: model selection is a lifecycle decision
balancing quality, safety, cost, and latency — not a default to "just use the
biggest managed endpoint."

## Open Questions for the Team

- Exact thresholds (`epsilon`, `citation_threshold`, `cost_ratio`) — set after a
  baseline frontier run so they're grounded in real numbers.
- LLM-as-judge model for correctness/groundedness scoring — which model, and how
  to avoid teacher-judging-student bias.
- Drift-slice window length (2 weeks? 4 weeks?) — depends on trace volume.
