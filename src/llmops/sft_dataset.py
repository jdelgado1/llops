"""Convert teacher traces -> OpenAI chat-completions SFT dataset.

The OpenAI `messages` format is consumed by BOTH Fireworks fine-tuning and
Foundry managed SFT, so this output is fine-tuning-home agnostic:

    {"messages": [
        {"role": "system",    "content": <analyst instructions>},
        {"role": "user",      "content": <question>},
        {"role": "assistant", "content": <teacher grounded answer>}
    ]}

Includes a light validator so we catch malformed rows before submitting a job.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ARTIFACTS = Path("artifacts")

SFT_SYSTEM = (
    "You are a TMG (Telco/Media/Gaming) market & competitive-intelligence "
    "research analyst. Answer the question with a concise, accurate, sourced "
    "briefing."
)


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
            question = (trace.get("question") or "").strip()
            answer = (trace.get("answer") or "").strip()
            if not question or not answer:
                continue
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SFT_SYSTEM},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )

    if not rows:
        raise RuntimeError(f"No usable traces found in {src}")

    _validate(rows)

    ARTIFACTS.mkdir(exist_ok=True)
    out = Path(out_path) if out_path else ARTIFACTS / f"sft-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} SFT examples (OpenAI chat format) -> {out}")
    return out, len(rows)


def _validate(rows: list[dict]) -> None:
    """Fail fast on malformed rows (what fine-tuning services reject)."""
    for i, r in enumerate(rows):
        msgs = r.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            raise ValueError(f"row {i}: 'messages' must be a list of >= 2 turns")
        if msgs[-1].get("role") != "assistant":
            raise ValueError(f"row {i}: last turn must be the assistant target")
        for m in msgs:
            if m.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"row {i}: invalid role {m.get('role')!r}")
            if not (m.get("content") or "").strip():
                raise ValueError(f"row {i}: empty content for role {m.get('role')!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an OpenAI-format SFT dataset from teacher traces.")
    ap.add_argument("--traces", required=True, help="path to traces JSONL from gen_traces")
    ap.add_argument("--out", default=None, help="output SFT JSONL path")
    args = ap.parse_args()
    build_sft(args.traces, args.out)


if __name__ == "__main__":
    main()
