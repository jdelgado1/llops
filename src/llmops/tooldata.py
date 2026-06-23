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
import re
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


def _load_from_hf(limit: int | None, categories: set[str] | None = None) -> list[ToolCallItem]:
    """Pull BFCL directly from Hugging Face repo files into our shape.

    BFCL is published as line-delimited ``.json`` files (not ``datasets`` splits)
    with ground-truth answers in ``possible_answer/``. We download and join by id.
    """
    from huggingface_hub import HfApi, hf_hub_download  # local import: only needed for hf

    def _iter_jsonl(path: Path):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _norm_messages(question) -> list[dict]:
        if isinstance(question, str):
            return [{"role": "user", "content": question}]
        if isinstance(question, list):
            out: list[dict] = []
            for turn in question:
                if isinstance(turn, list):
                    out.extend([m for m in turn if isinstance(m, dict)])
                elif isinstance(turn, dict):
                    out.append(turn)
            return out
        return []

    def _safe_fn_name(name: str) -> str:
        name = re.sub(r"[^A-Za-z0-9_-]", "_", name or "")
        if not name:
            name = "tool_fn"
        return name[:64]

    def _sanitize_schema(node):
        allowed = {"object", "string", "number", "integer", "boolean", "array", "null"}
        if not isinstance(node, dict):
            return {"type": "string"}

        out = dict(node)
        t = out.get("type")
        if t == "dict":
            t = "object"
        if isinstance(t, str) and t not in allowed:
            # BFCL language-specific signatures often use non-JSON-schema types.
            t = "string"
        if t:
            out["type"] = t

        if out.get("type") == "object":
            props = out.get("properties")
            if not isinstance(props, dict):
                props = {}
            out["properties"] = {k: _sanitize_schema(v) for k, v in props.items()}
            req = out.get("required")
            out["required"] = req if isinstance(req, list) else []

        if out.get("type") == "array":
            out["items"] = _sanitize_schema(out.get("items") or {"type": "string"})

        for comb in ("anyOf", "oneOf", "allOf"):
            if comb in out and isinstance(out[comb], list):
                out[comb] = [_sanitize_schema(x) for x in out[comb] if isinstance(x, dict)]

        return out

    def _norm_tools(functions) -> tuple[list[dict], dict[str, str]]:
        out = []
        name_map: dict[str, str] = {}
        for fn in (functions or []):
            if not isinstance(fn, dict):
                continue
            wrapped = fn if fn.get("type") == "function" else {"type": "function", "function": fn}
            f = dict((wrapped.get("function") or {}))
            orig_name = str(f.get("name") or "")
            safe_name = _safe_fn_name(orig_name)
            name_map[orig_name] = safe_name
            f["name"] = safe_name
            f["parameters"] = _sanitize_schema(f.get("parameters") or {"type": "object", "properties": {}})
            out.append({"type": "function", "function": f})
        return out, name_map

    def _norm_reference(ground_truth, name_map: dict[str, str]) -> list[dict]:
        refs: list[dict] = []
        for call in (ground_truth or []):
            if not isinstance(call, dict):
                continue
            for name, args in call.items():
                safe_name = name_map.get(name, _safe_fn_name(name))
                norm_args = {}
                if isinstance(args, dict):
                    for k, v in args.items():
                        norm_args[k] = v if isinstance(v, list) else [v]
                refs.append({"name": safe_name, "arguments": norm_args})
        return refs

    api = HfApi()
    files = api.list_repo_files(HF_DATASET, repo_type="dataset")

    # Use only tasks with paired possible answers (objective AST references).
    answer_files = {
        p.split("possible_answer/", 1)[1]
        for p in files
        if p.startswith("possible_answer/BFCL_v3_") and p.endswith(".json")
    }
    data_files = sorted(
        [
            p
            for p in files
            if p.startswith("BFCL_v3_")
            and p.endswith(".json")
            and "/" not in p
            and p in answer_files
        ]
    )
    if not data_files:
        raise RuntimeError(f"Could not find paired BFCL data/answer files in {HF_DATASET}.")

    if categories:
        data_files = [
            p for p in data_files
            if p.removeprefix("BFCL_v3_").removesuffix(".json") in categories
        ]
        if not data_files:
            raise RuntimeError(
                f"No BFCL files match categories={categories}. "
                f"Available example cats: simple, multiple, parallel, parallel_multiple, "
                f"live_simple, live_multiple, live_parallel, live_parallel_multiple."
            )

    items: list[ToolCallItem] = []
    for data_name in data_files:
        cat = data_name.removeprefix("BFCL_v3_").removesuffix(".json")
        data_path = Path(
            hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=data_name)
        )
        ans_path = Path(
            hf_hub_download(
                repo_id=HF_DATASET,
                repo_type="dataset",
                filename=f"possible_answer/{data_name}",
            )
        )

        raw_answers_by_id = {
            str(r.get("id") or "").strip(): (r.get("ground_truth") or [])
            for r in _iter_jsonl(ans_path)
        }

        for row in _iter_jsonl(data_path):
            tid = str(row.get("id") or "").strip()
            if not tid:
                continue
            messages = _norm_messages(row.get("question") or row.get("messages"))
            tools, name_map = _norm_tools(row.get("function") or row.get("tools"))
            reference = _norm_reference(raw_answers_by_id.get(tid) or [], name_map)
            if not messages or not tools or not reference:
                continue
            items.append(
                ToolCallItem(
                    tid=tid,
                    category=cat,
                    messages=messages,
                    tools=tools,
                    reference=reference,
                )
            )
            if limit is not None and len(items) >= limit:
                return items

    if not items:
        raise RuntimeError(
            f"Could not parse HF BFCL repo {HF_DATASET}; use TOOLCALLING_SOURCE=sample only as fallback."
        )
    return items


def load_toolcalling(
    source: str = "sample",
    eval_pct: int = 30,
    limit: int | None = None,
    categories: set[str] | None = None,
) -> tuple[list[ToolCallItem], list[ToolCallItem]]:
    """Load tool-calling items and split into (trace_pool, eval_pool).

    Args:
        source: ``sample`` | ``hf`` | a filesystem path to JSONL data.
        eval_pct: percent of items held out for evaluation (never trained on).
        limit: optional cap on total items (handy for quick smoke runs).
        categories: optional set of BFCL categories to keep (hf source only),
            e.g. {"parallel_multiple", "parallel"} for a harder eval.
    """
    if source == "hf":
        items = _load_from_hf(limit, categories)
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
