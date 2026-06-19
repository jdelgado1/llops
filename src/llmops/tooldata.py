"""Tool-calling datasets and the by-id train/eval split.

HARD RULE (no test-set leakage): we never train on eval items. The source is
split *by item id* into:
  - a trace-generation pool  -> teacher emits tool calls -> (rejection-sampled) training data
  - a held-out eval pool      -> only ever used to score AST accuracy / gate

The split is deterministic (hash of the item id), so it's stable across runs.

Data shape (BFCL-style):
    ToolCallItem(
        tid,            # stable id
        category,       # simple | multiple | parallel | parallel_multiple
        messages,       # OpenAI chat messages (the user request)
        tools,          # OpenAI tool schemas offered to the model
        reference,      # list of ground-truth calls; each arg -> list of acceptable values
    )

Sources (``TOOLCALLING_SOURCE`` / ``--source``):
  - ``sample`` (default): the bundled offline set in ``data/toolcalling_sample.jsonl``
    — TMG ops/support tools, runs anywhere with no network.
  - a filesystem path: a directory or single ``.jsonl`` of the same shape
    (e.g. exported BFCL / ToolACE data).
  - ``hf``: pull the Berkeley Function-Calling Leaderboard from Hugging Face at
    scale (best-effort; schema-defensive).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "toolcalling_sample.jsonl"
HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


@dataclass
class ToolCallItem:
    tid: str
    category: str
    messages: list[dict]
    tools: list[dict]
    reference: list[dict] = field(default_factory=list)

    def user_text(self) -> str:
        """First user turn, for compact logging."""
        for m in self.messages:
            if m.get("role") == "user":
                return (m.get("content") or "").strip()
        return ""


def _eval_bucket(tid: str, buckets: int = 100) -> int:
    digest = hashlib.sha256(tid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def _item_from_row(row: dict) -> ToolCallItem | None:
    tid = str(row.get("tid") or row.get("id") or "").strip()
    tools = row.get("tools") or []
    messages = row.get("messages") or []
    reference = row.get("reference") or []
    if not tid or not tools or not messages:
        return None
    return ToolCallItem(
        tid=tid,
        category=str(row.get("category") or "simple"),
        messages=messages,
        tools=tools,
        reference=reference,
    )


def _load_jsonl(path: Path) -> list[ToolCallItem]:
    items: list[ToolCallItem] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = _item_from_row(json.loads(line))
            if it:
                items.append(it)
    return items


def _load_from_path(path: Path) -> list[ToolCallItem]:
    if path.is_dir():
        items: list[ToolCallItem] = []
        for p in sorted(path.glob("*.jsonl")):
            items.extend(_load_jsonl(p))
        return items
    return _load_jsonl(path)


def _load_from_hf(limit: int | None) -> list[ToolCallItem]:
    """Best-effort pull of BFCL from Hugging Face into our shape.

    The HF mirror's exact schema varies by revision; we read defensively and
    raise a clear error if the expected fields aren't present so the caller can
    fall back to ``sample`` or a local export.
    """
    from datasets import load_dataset  # local import: only needed for hf

    ds = load_dataset(HF_DATASET, split="train")
    items: list[ToolCallItem] = []
    for row in ds:
        question = row.get("question") or row.get("messages")
        functions = row.get("function") or row.get("tools")
        if not question or not functions:
            continue
        # normalize functions -> OpenAI tool schemas
        tools = []
        for fn in functions:
            if isinstance(fn, dict) and fn.get("type") == "function":
                tools.append(fn)
            elif isinstance(fn, dict):
                tools.append({"type": "function", "function": fn})
        # normalize messages
        if isinstance(question, str):
            messages = [{"role": "user", "content": question}]
        else:
            messages = [m for turn in question for m in (turn if isinstance(turn, list) else [turn])]
        it = ToolCallItem(
            tid=str(row.get("id") or len(items)),
            category=str(row.get("category") or "simple"),
            messages=messages,
            tools=tools,
            reference=row.get("ground_truth") or row.get("reference") or [],
        )
        items.append(it)
        if limit is not None and len(items) >= limit:
            break
    if not items:
        raise RuntimeError(
            f"Could not parse {HF_DATASET}; set TOOLCALLING_SOURCE=sample or a local path."
        )
    return items


def load_toolcalling(
    source: str = "sample",
    eval_pct: int = 30,
    limit: int | None = None,
) -> tuple[list[ToolCallItem], list[ToolCallItem]]:
    """Load tool-calling items and split into (trace_pool, eval_pool).

    Args:
        source: ``sample`` | ``hf`` | a filesystem path to JSONL data.
        eval_pct: percent of items held out for evaluation (never trained on).
        limit: optional cap on total items (handy for quick smoke runs).
    """
    if source == "hf":
        items = _load_from_hf(limit)
    elif source == "sample":
        items = _load_jsonl(SAMPLE_PATH)
    else:
        items = _load_from_path(Path(source))

    if limit is not None:
        items = items[:limit]
    if not items:
        raise RuntimeError(f"No tool-calling items loaded from source={source!r}.")

    trace_pool: list[ToolCallItem] = []
    eval_pool: list[ToolCallItem] = []
    for it in items:
        (eval_pool if _eval_bucket(it.tid) < eval_pct else trace_pool).append(it)
    return trace_pool, eval_pool
