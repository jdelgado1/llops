# Scenario — TMG Operations & Support Tool-Calling Agent

> Task #1. The mock customer scenario the whole demo is built around. Keep it
> concrete and legible for an infra/data audience.

## The customer

A **Telco / Media / Gaming (TMG)** operator runs an internal **operations &
support agent** that frontline staff and automated workflows use to act on
customer and network issues. Instead of writing prose, the agent's job is to
**call the right internal tool with the right arguments** — open a ticket, check
a network status, reset a modem, look up a game server, schedule a technician,
or search the live web for competitive/outage context.

This is a realistic agent workload: the value is in **taking the correct action**
(the correct function call), not in writing a paragraph.

## What the agent does (one turn)

**Input:** a user request + the set of tool schemas currently available.

**Output:** the **correct function call(s)** — the right tool name and a correct,
fully-typed argument object.

Example:

> **Request:** "Open a high priority support ticket for customer CU-9012 about a
> billing error."
>
> **Correct call:**
> `create_support_ticket(customer_id="CU-9012", issue_type="billing", priority="high")`

The agent may also need to:

- **disambiguate among tools** (pick `reset_modem` vs `schedule_technician`),
- **emit parallel calls** (check network status for two regions at once), and
- **invoke `web_search`** (Microsoft Web IQ) when the answer needs live web
  context — this is how the "web search" thread from the agenda is honored,
  *as a tool*, without making fuzzy prose the task.

## The tools (bundled TMG set)

The bundled [`data/toolcalling_sample.jsonl`](../data/toolcalling_sample.jsonl)
defines a small, legible TMG toolset:

| Tool | Purpose |
| --- | --- |
| `get_network_status(region, service_type)` | Mobile/broadband/fiber status |
| `check_data_usage(account_id, billing_cycle?)` | Account data usage |
| `create_support_ticket(customer_id, issue_type, priority)` | Open a ticket |
| `escalate_ticket(ticket_id, tier)` | Escalate a ticket |
| `reset_modem(device_id)` | Remote modem reset |
| `schedule_technician(account_id, date, window)` | Book an on-site visit |
| `lookup_game_server(game, region)` | Game server cluster status |
| `get_streaming_outage(platform, region)` | Streaming outage check |
| `get_subscription(account_id)` | Subscription plan lookup |
| `web_search(query, freshness?)` | Live web (Web IQ) lookup |

At scale, swap in **ToolACE** (richer training prompts/schemas) and **BFCL**
(the standard eval) via `TOOLCALLING_SOURCE`.

## Who uses it

- **Frontline support / NOC staff** — act on tickets and outages fast.
- **Automated workflows** — the agent is the "hands" that call backend APIs.
- **PMs / analysts** — use the `web_search` tool for competitive/outage context.

## Why this drifts (why you retrain)

- **Tool schemas change:** new tools ship; arguments/enums change; deprecated
  tools are removed. A model trained on the old schemas starts calling things
  wrong → **AST accuracy drops**.
- **Usage mix shifts:** seasonal events (game launches, sports seasons, outages)
  change which tools dominate.
- **New entities:** new device types, plans, platforms, regions.

Because the **action surface itself drifts**, the agent must be **retrained on
fresh correct traces** — which is exactly the lifecycle story. Retraining is
visibly justified: change the tools, watch the score fall, retrain, watch it
recover.

## Why this scenario (vs grounded research QA)

We first built a grounded research-QA scenario; on Foundry it **saturated**
(frontier = base = distilled ≈ 90%, distillation 0 lift). Tool calling, scored by
**objective AST accuracy**, is the task that actually demonstrates **a small
distilled model matching or beating a frontier model** — the headline the session
needs. See `CONTEXT.md` for the full pivot rationale.
