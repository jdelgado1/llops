"""Call a Foundry deployment with tools and parse the predicted tool call(s).

Works for both GPT and qwen3 deployments via the OpenAI-compatible
``chat.completions`` ``tools`` interface. If a model returns its call(s) as text
instead of structured ``tool_calls`` (some OSS models do, e.g. Hermes-style
``<tool_call>{...}</tool_call>``), we parse them out as a fallback so scoring is
fair across model families.

Returns ``(calls, usage, latency_s)`` where ``calls`` is a list of
``{"name": str, "arguments": dict}`` — the shape the AST checker expects.
"""
from __future__ import annotations

import json
import re
import time

# qwen3 reasoning trace; strip before parsing any text-form tool calls.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOLCALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

TOOL_SYSTEM = (
    "You are a TMG (Telco/Media/Gaming) operations & support agent. Use the "
    "provided tools to fulfill the user's request. Call the correct function(s) "
    "with correct arguments. Do not ask follow-up questions if the request is "
    "answerable with the tools."
)


def _strip_think(text: str | None) -> str:
    return _THINK_RE.sub("", text or "").strip()


def _parse_calls_from_text(content: str | None) -> list[dict]:
    """Best-effort extraction of tool calls embedded in free text."""
    text = _strip_think(content)
    calls: list[dict] = []
    for m in _TOOLCALL_TAG_RE.finditer(text):
        with _suppress():
            obj = json.loads(m.group(1))
            name = obj.get("name") or obj.get("function")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                calls.append({"name": name, "arguments": args})
    if calls:
        return calls
    # last resort: a bare JSON object with name/arguments
    with _suppress():
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        name = obj.get("name") or obj.get("function")
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            args = json.loads(args)
        if name:
            calls.append({"name": name, "arguments": args})
    return calls


class _suppress:
    """Tiny inline 'suppress all exceptions' context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


def _messages_with_system(messages: list[dict]) -> list[dict]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": TOOL_SYSTEM}, *messages]


def call_with_tools(
    client,
    model: str,
    messages: list[dict],
    tools: list[dict],
    tool_choice: str = "auto",
) -> tuple[list[dict], object, float]:
    """Invoke ``model`` with ``tools`` and return parsed calls + usage + latency.

    ``tool_choice``: ``auto`` (model decides — used for eval) or ``required``
    (force at least one call — used for teacher trace generation).
    """
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=_messages_with_system(messages),
        tools=tools,
        tool_choice=tool_choice,
        max_completion_tokens=2048,
    )
    latency_s = time.perf_counter() - t0

    msg = resp.choices[0].message
    calls: list[dict] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        if not fn:
            continue
        try:
            args = json.loads(fn.arguments or "{}")
        except (ValueError, TypeError):
            args = {}
        calls.append({"name": fn.name, "arguments": args})

    if not calls:
        calls = _parse_calls_from_text(getattr(msg, "content", None))

    return calls, getattr(resp, "usage", None), latency_s
