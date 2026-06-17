# Golden Dataset & Evaluation Metrics

> Task #2 from Nikhil's list: *Identify golden dataset and eval metric to
> benchmark frontier model vs smaller OSS model.*

## Design Principle (Evaluate on benchmarks, train on traces)

The single most important rule in this design: **we evaluate the model on public
benchmark questions, but we never train on them.** The student is trained *only*
on **frontier + web-search (WebIQ) teacher traces**. Mixing the two would leak the test set
into the training loop and turn the headline metric into a lie.

So the data splits into two cleanly separated roles:

1. **Evaluation layer — public HF benchmarks (never trained on).** Ready-made
   **questions + reference answers** give us a reproducible, credible scoreboard.
   We use *several* benchmarks at different jobs (frozen regression, temporal
   drift, hard ceiling) — see
   [Dataset Architecture](#dataset-architecture-layered).
2. **Training layer — frontier teacher traces (the only thing we train on).**
   The frontier model (the **teacher**) answers questions *with web search
   (WebIQ) grounding*;
   its high-quality, reviewed completions `(question + retrieved_context →
   answer + citations)` become the **distillation training corpus**. This matches
   the agenda's "capture prompts, retrieval context, completions, citations,
   human ratings" step.

> **In plain terms:** public benchmarks are the *exam* (we only grade on them);
> the frontier model's WebIQ-grounded answers are the *study material* (the only
> thing the student learns from). Keeping the exam out of the study material is
> what makes the score honest.

### Train/Eval Separation (Hard Rule)

- Fresh questions are partitioned **by question** into a **trace-generation
  pool** (used to make teacher training data) and a **held-out eval pool** (used
  to score + gate). The two never overlap.
- The **frozen regression set is never trained on, ever.**
- Promotion is decided on **held-out fresh questions** the student has never seen
  traces for.

### Why retrain at all, if grounding is live?

The obvious heckle from an infra/data audience: *"if it searches the web every
time, why would it ever go stale?"* The answer is precise and worth stating
plainly:

We distill **synthesis and citation over the provided context — not facts.**
Retrieval stays live via web search (WebIQ), so the student doesn't memorize (or forget)
*facts*. What drifts is the **distribution**: new entities and terminology,
false-premise traps, changed evidence/source formats, and shifts in the mix of
questions users ask. The learned *synthesizer* misfires as that distribution
moves, the Answer Quality Score drops, and retraining on **recent teacher
traces** re-aligns it to the current world. (We deliberately do *not* bake facts
into the weights — that would fight the "grounding does the work" thesis and
invite stale hallucinations.)

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

## Dataset Architecture (Layered)

Rather than one seed dataset, we use a small **stack of public benchmarks**, each
doing a different job. All are evaluation-only; none are trained on.

| Layer | Dataset | Role | Verified facts |
| --- | --- | --- | --- |
| **Frozen regression** | **`aialt/RetrievalQA`** | Stable, apples-to-apples "does the student still produce good grounded answers?" | MIT license; 2,785 short-form open-domain Qs; fields `question`, `ground_truth` (list), `context` (retrieved evidence `title`+`text`), and a **`param_knowledge_answerable`** flag marking the **1,271 questions that *require* external retrieval** vs 1,514 answerable from memory |
| **Temporal drift eval** | **FreshQA** (dated snapshots) | "The world changed" — fast/slow/never-changing facts + false-premise traps | Re-released in **dated community snapshots** (`natyou/freshqa_10_06`, `bojanbabic/freshqa_072825`, `…_08182025`), 600 Qs each → real drift across windows, not faked |
| **Replayable weekly stream** *(optional)* | **RealTimeQA** (mirror) | Simulated new production queries arriving over time | Canonical `realtimeqa/realtimeqa_public` is **gated (401)**; use a mirror such as `SKIML-ICL/CRRAG_realtimeqa` (~100 rows). It's a **replayable timestamped archive**, not a live feed — verify format at load time |
| **Hard ceiling** | **`OpenResearcher/web-bench`** | Shows the frontier-vs-distilled **gap / headroom** on genuinely hard web research | Apache-2.0; unified `query_id`/`question`/`answer`; use clean English splits **`webwalkerqa_ref`, `seal_ref`, `gaia_text`**. **Skip `hle`** (multiple-choice → reintroduces the banned classification flavor) and the **encrypted** `browsecomp` / `xbench` splits unless you accept the extra setup |
| **Training corpus** | **Frontier + web-search (WebIQ) traces** | The *only* thing the student is distilled on | Generated by us (see [Train/Eval Separation](#traineval-separation-hard-rule)) |
| **TMG demo slice** *(presentation)* | Custom TMG prompts | Business-flavored example vertical for the talk | Hand-picked telco/media/gaming current-events prompts; a presentation veneer, kept out of the rigorous eval |

> **The LLMOps story this enables:** *the frozen regression eval proves the model
> still works; the fresh eval proves the world changed; the drift trigger proves
> why retraining matters.*

### Why these and not FinanceBench / a single seed

The earlier proposal (`PatronusAI/financebench`) was **dropped**: it's QA over
**static PDF filings**, so live web search would be cosmetic and there is no real
drift to retrain against. The benchmarks above make grounding **necessary**
(`RetrievalQA`'s retrieval flag) and drift **real** (`FreshQA`'s dated
snapshots) — the two properties the loop depends on. `DailyQA` was considered but
**does not exist on HF** (verified); `NewsQA` and `TeleQnA` were rejected (span /
multiple-choice formats respectively).

> Answers across these sets are mostly **short and verifiable**, which is a
> *feature*: it keeps the single Answer Quality Score clean to compute. The
> briefing/house *voice* comes from the frontier teacher's output style, not from
> the benchmark answers.

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
| `retrieved_context` | Web search (WebIQ) results (+ any tool outputs), **snapshotted at trace time** |
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

Mapping the [layered architecture](#dataset-architecture-layered) to concrete
sets:

- **Evaluation sets (never trained on):**
  - **Frozen regression** — `RetrievalQA` (stable; apples-to-apples over time).
  - **Drift slice** — the current `FreshQA` snapshot (and/or a replayed
    `RealTimeQA` window), **held out** to surface quality decay.
  - **Hard ceiling** — selected `web-bench` splits to show frontier-vs-distilled
    headroom.
  - **Held-out fresh promotion set** — fresh questions the student has *no*
    teacher traces for; the gate is decided here.
- **Training set (the only trained-on data):** reviewer-accepted **frontier +
  web-search (WebIQ) traces** `(question + retrieved_context → answer + citations)`,
  drawn from the **trace-generation pool only** (disjoint from every eval set).
- **TMG demo slice (presentation only):** hand-picked telco/media/gaming
  current-events prompts for the talk — never mixed into the scored eval.

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
- Which `RealTimeQA` mirror to standardize on (canonical repo is gated) and the
  replay window length per "weekly" step.
- Drift-slice window length (2 weeks? 4 weeks?) — depends on trace volume.
