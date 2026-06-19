"""Convert teacher tool-call traces -> OpenAI function-calling SFT dataset.

Foundry managed SFT (and OpenAI fine-tuning) consume the chat ``messages``
format with a tool-calling assistant target:

    {"messages": [
        {"role": "system", "content": <agent instructions>},
        {"role": "user",   "content": <request>},
        {"role": "assistant", "content": null,
         "tool_calls": [{"id": "...", "type": "function",
                          "function": {"name": "...", "arguments": "<json string>"}}]}
     ],
     "tools": [ <the tool schemas offered> ]}

So the student learns *which* function to call and *with what arguments* — the
behavior we score with AST accuracy. Includes a validator so we catch malformed
rows before submitting a Foundry fine-tune job.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .tool_models import TOOL_SYSTEM

ARTIFACTS = Path("artifacts")


def _assistant_message(calls: list[dict]) -> dict:
    tool_calls = []
    for i, c in enumerate(calls):
        tool_calls.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False),
                },
            }
        )
    return {"role": "assistant", "tool_calls": tool_calls}


def build_sft(traces_path: str, out_path: str | None = None) -> tuple[Path, int]:
    src = Path(traces_path)
    if not src.exists():
        raise FileNotFoundError(src)

    rows: list[dict] = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trace = json.loads(line)
            calls = trace.get("tool_calls") or []
            messages = trace.get("messages") or []
            tools = trace.get("tools") or []
            if not calls or not messages or not tools:
                continue
            sys_msgs = [m for m in messages if m.get("role") == "system"]
            convo = [m for m in messages if m.get("role") != "system"]
            system = sys_msgs[0] if sys_msgs else {"role": "system", "content": TOOL_SYSTEM}
            rows.append(
                {
                    "messages": [system, *convo, _assistant_message(calls)],
                    "tools": tools,
                }
            )

    if not rows:
        raise RuntimeError(f"No usable tool-call traces found in {src}")

    _validate(rows)

    ARTIFACTS.mkdir(exist_ok=True)
    out = Path(out_path) if out_path else ARTIFACTS / f"tool-sft-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} function-calling SFT examples -> {out}")
    return out, len(rows)


def _validate(rows: list[dict]) -> None:
    """Fail fast on malformed rows (what fine-tuning services reject)."""
    for i, r in enumerate(rows):
        msgs = r.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            raise ValueError(f"row {i}: 'messages' must be a list of >= 2 turns")
        if not isinstance(r.get("tools"), list) or not r["tools"]:
            raise ValueError(f"row {i}: 'tools' must be a non-empty list")
        target = msgs[-1]
        if target.get("role") != "assistant":
            raise ValueError(f"row {i}: last turn must be the assistant target")
        tcs = target.get("tool_calls")
        if not isinstance(tcs, list) or not tcs:
            raise ValueError(f"row {i}: assistant target must have tool_calls")
        for tc in tcs:
            fn = tc.get("function") or {}
            if not fn.get("name"):
                raise ValueError(f"row {i}: tool_call missing function name")
            try:
                json.loads(fn.get("arguments") or "{}")
            except (ValueError, TypeError):
                raise ValueError(f"row {i}: tool_call arguments not valid JSON string")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a function-calling SFT dataset from teacher tool-call traces.")
    ap.add_argument("--traces", required=True, help="path to tool-call traces JSONL from gen_tool_traces")
    ap.add_argument("--out", default=None, help="output SFT JSONL path")
    args = ap.parse_args()
    build_sft(args.traces, args.out)


if __name__ == "__main__":
    main()
