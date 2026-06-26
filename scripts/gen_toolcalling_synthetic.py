"""Generate a larger local BFCL-style tool-calling dataset.

This is a fallback when ``TOOLCALLING_SOURCE=hf`` is unavailable or schema-drifts.
It writes rows in the same shape as ``data/toolcalling_sample.jsonl`` so the
existing split/trace/eval/SFT pipeline can be reused as-is.

Example:
    python scripts/gen_toolcalling_synthetic.py --out data/toolcalling_synthetic.jsonl --rows 360
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import date, timedelta
from pathlib import Path


def _tool(name: str, description: str, properties: dict, required: list[str], enums: dict | None = None) -> dict:
    props = copy.deepcopy(properties)
    if enums:
        for key, vals in enums.items():
            if key in props:
                props[key]["enum"] = vals
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


TOOLS = {
    "get_network_status": _tool(
        "get_network_status",
        "Get the current network status for a service in a region.",
        {"region": {"type": "string"}, "service_type": {"type": "string"}},
        ["region", "service_type"],
        enums={"service_type": ["mobile", "broadband", "fiber"]},
    ),
    "check_data_usage": _tool(
        "check_data_usage",
        "Look up data usage for an account.",
        {"account_id": {"type": "string"}, "billing_cycle": {"type": "string"}},
        ["account_id"],
        enums={"billing_cycle": ["current", "previous"]},
    ),
    "create_support_ticket": _tool(
        "create_support_ticket",
        "Create a customer support ticket.",
        {
            "customer_id": {"type": "string"},
            "issue_type": {"type": "string"},
            "priority": {"type": "string"},
        },
        ["customer_id", "issue_type", "priority"],
        enums={
            "issue_type": ["billing", "network", "streaming", "hardware"],
            "priority": ["low", "medium", "high"],
        },
    ),
    "reset_modem": _tool(
        "reset_modem",
        "Remotely reset a customer modem.",
        {"device_id": {"type": "string"}},
        ["device_id"],
    ),
    "lookup_game_server": _tool(
        "lookup_game_server",
        "Look up the status of a game server cluster.",
        {"game": {"type": "string"}, "region": {"type": "string"}},
        ["game", "region"],
        enums={"region": ["NA", "EU", "APAC", "SA"]},
    ),
    "get_streaming_outage": _tool(
        "get_streaming_outage",
        "Check whether a streaming platform is experiencing an outage.",
        {"platform": {"type": "string"}, "region": {"type": "string"}},
        ["platform", "region"],
    ),
    "escalate_ticket": _tool(
        "escalate_ticket",
        "Escalate an existing support ticket to a higher support tier.",
        {"ticket_id": {"type": "string"}, "tier": {"type": "integer"}},
        ["ticket_id", "tier"],
        enums={"tier": [1, 2, 3]},
    ),
    "schedule_technician": _tool(
        "schedule_technician",
        "Schedule an on-site technician visit.",
        {"account_id": {"type": "string"}, "date": {"type": "string"}, "window": {"type": "string"}},
        ["account_id", "date", "window"],
        enums={"window": ["morning", "afternoon", "evening"]},
    ),
    "get_subscription": _tool(
        "get_subscription",
        "Get the subscription plan details for an account.",
        {"account_id": {"type": "string"}},
        ["account_id"],
    ),
    "web_search": _tool(
        "web_search",
        "Search the live web for current information.",
        {"query": {"type": "string"}, "freshness": {"type": "string"}},
        ["query"],
        enums={"freshness": ["day", "week", "month"]},
    ),
}


def _row(tid: str, category: str, user: str, tools: list[str], reference: list[dict]) -> dict:
    return {
        "tid": tid,
        "category": category,
        "messages": [{"role": "user", "content": user}],
        "tools": [copy.deepcopy(TOOLS[t]) for t in tools],
        "reference": reference,
    }


def _build(rows: int, seed: int) -> list[dict]:
    r = random.Random(seed)
    cities = ["Dallas", "Austin", "Houston", "Phoenix", "Seattle", "Miami", "Denver", "Atlanta"]
    service_types = ["mobile", "broadband", "fiber"]
    issue_types = ["billing", "network", "streaming", "hardware"]
    priorities = ["low", "medium", "high"]
    games = ["Apex Legends", "Valorant", "Fortnite", "Rocket League", "Call of Duty"]
    game_regions = ["NA", "EU", "APAC", "SA"]
    platforms = ["StreamMax", "CinemaNow", "CloudTV", "PlayFlix"]
    world_regions = ["North America", "Europe", "Asia", "South America"]
    freshness = ["day", "week", "month"]
    windows = ["morning", "afternoon", "evening"]

    out: list[dict] = []
    d0 = date(2026, 7, 1)

    i = 0
    while len(out) < rows:
        kind = i % 10
        i += 1

        if kind == 0:
            city = r.choice(cities)
            svc = r.choice(service_types)
            out.append(
                _row(
                    f"syn_net_{i}",
                    "simple",
                    f"Is there a {svc} outage in {city} right now?",
                    ["get_network_status"],
                    [{"name": "get_network_status", "arguments": {"region": [city], "service_type": [svc]}}],
                )
            )
        elif kind == 1:
            acct = f"AC-{r.randint(1000, 9999)}"
            cyc = r.choice(["current", "previous"])
            out.append(
                _row(
                    f"syn_usage_{i}",
                    "simple",
                    f"Check {cyc} billing cycle data usage for account {acct}.",
                    ["check_data_usage"],
                    [{"name": "check_data_usage", "arguments": {"account_id": [acct], "billing_cycle": [cyc, ""]}}],
                )
            )
        elif kind == 2:
            cust = f"CU-{r.randint(1000, 9999)}"
            issue = r.choice(issue_types)
            pri = r.choice(priorities)
            out.append(
                _row(
                    f"syn_ticket_{i}",
                    "multiple",
                    f"Create a {pri} priority {issue} support ticket for customer {cust}.",
                    ["create_support_ticket", "get_network_status", "check_data_usage"],
                    [
                        {
                            "name": "create_support_ticket",
                            "arguments": {"customer_id": [cust], "issue_type": [issue], "priority": [pri]},
                        }
                    ],
                )
            )
        elif kind == 3:
            dev = f"DV-{r.randint(10, 999)}"
            out.append(
                _row(
                    f"syn_modem_{i}",
                    "multiple",
                    f"Customer internet is down. Please reset modem {dev}.",
                    ["reset_modem", "schedule_technician", "get_network_status"],
                    [{"name": "reset_modem", "arguments": {"device_id": [dev]}}],
                )
            )
        elif kind == 4:
            game = r.choice(games)
            reg = r.choice(game_regions)
            out.append(
                _row(
                    f"syn_game_{i}",
                    "multiple",
                    f"What is the server status for {game} in {reg}?",
                    ["lookup_game_server", "get_streaming_outage", "get_network_status"],
                    [{"name": "lookup_game_server", "arguments": {"game": [game], "region": [reg]}}],
                )
            )
        elif kind == 5:
            plat = r.choice(platforms)
            reg = r.choice(world_regions)
            out.append(
                _row(
                    f"syn_stream_{i}",
                    "multiple",
                    f"Is {plat} down in {reg}?",
                    ["get_streaming_outage", "lookup_game_server"],
                    [{"name": "get_streaming_outage", "arguments": {"platform": [plat], "region": [reg]}}],
                )
            )
        elif kind == 6:
            t = f"TK-{r.randint(100, 999)}"
            tier = r.choice([2, 3])
            out.append(
                _row(
                    f"syn_escalate_{i}",
                    "multiple",
                    f"Escalate ticket {t} to tier {tier}.",
                    ["escalate_ticket", "create_support_ticket"],
                    [{"name": "escalate_ticket", "arguments": {"ticket_id": [t], "tier": [tier]}}],
                )
            )
        elif kind == 7:
            acct = f"AC-{r.randint(10, 999)}"
            dt = (d0 + timedelta(days=r.randint(0, 90))).isoformat()
            w = r.choice(windows)
            out.append(
                _row(
                    f"syn_sched_{i}",
                    "multiple",
                    f"Schedule a technician for account {acct} on {dt} in the {w} window.",
                    ["schedule_technician", "reset_modem"],
                    [{"name": "schedule_technician", "arguments": {"account_id": [acct], "date": [dt], "window": [w]}}],
                )
            )
        elif kind == 8:
            a1 = f"AC-{r.randint(1, 499)}"
            a2 = f"AC-{r.randint(500, 999)}"
            out.append(
                _row(
                    f"syn_parallel_usage_{i}",
                    "parallel",
                    f"Pull current billing-cycle data usage for accounts {a1} and {a2}.",
                    ["check_data_usage"],
                    [
                        {"name": "check_data_usage", "arguments": {"account_id": [a1], "billing_cycle": ["current", ""]}},
                        {"name": "check_data_usage", "arguments": {"account_id": [a2], "billing_cycle": ["current", ""]}},
                    ],
                )
            )
        else:
            q = r.choice(
                [
                    "reported 5G fixed wireless outages by competitors",
                    "major cloud gaming downtime incidents",
                    "regional broadband outage news",
                    "large telecom service disruption reports",
                ]
            )
            fresh = r.choice(freshness)
            out.append(
                _row(
                    f"syn_web_{i}",
                    "multiple",
                    f"Search the web for {q} this {fresh}.",
                    ["web_search", "get_network_status"],
                    [{"name": "web_search", "arguments": {"query": [q], "freshness": [fresh, ""]}}],
                )
            )

    return out[:rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic BFCL-style tool-calling data.")
    ap.add_argument("--out", default="data/toolcalling_synthetic.jsonl")
    ap.add_argument("--rows", type=int, default=360)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = _build(rows=args.rows, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
