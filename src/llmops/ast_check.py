"""The single headline metric: **AST accuracy** for tool/function calling.

This is the BFCL (Berkeley Function-Calling Leaderboard) "AST" idea, implemented
in-repo so the metric is legible and self-contained (no external eval harness):

    A predicted tool call is CORRECT iff
      1. it names the right function,
      2. every REQUIRED parameter is present,
      3. each parameter's value is one of the reference's ACCEPTABLE values
         (BFCL ground-truth allows several valid values per argument), and
      4. it invents no parameter the schema doesn't define.

For parallel / multiple-call categories, the prediction is a *list* of calls and
is correct iff it matches the reference list as a set (same count, each reference
call matched by a distinct predicted call).

`AST accuracy` for a model = % of eval items whose predicted call(s) are correct.
One objective number, no LLM-judge bias — that's the scoreboard.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CallResult:
    correct: bool
    reason: str


# Reference / acceptable values. BFCL encodes optional-but-omittable arguments by
# including "" (or null) among the acceptable values.
_OPTIONAL_SENTINELS = ("", None)


def _coerce_number(value):
    """Best-effort numeric coercion so 5, 5.0, and "5" compare equal."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return value
    return value


def _scalar_equal(pred, acceptable) -> bool:
    """Type-tolerant scalar equality (handles num-as-string, case, whitespace)."""
    if pred == acceptable:
        return True
    pn, an = _coerce_number(pred), _coerce_number(acceptable)
    if isinstance(pn, (int, float)) and isinstance(an, (int, float)):
        return float(pn) == float(an)
    if isinstance(pred, str) and isinstance(acceptable, str):
        return pred.strip().lower() == acceptable.strip().lower()
    return False


def _value_matches(pred, acceptable_values) -> bool:
    """True if ``pred`` equals any value in the reference's acceptable list."""
    if not isinstance(acceptable_values, list):
        acceptable_values = [acceptable_values]
    for a in acceptable_values:
        if isinstance(a, list) and isinstance(pred, list):
            # list-valued argument: order-insensitive element match
            if len(a) == len(pred) and all(
                any(_scalar_equal(p, x) for x in a) for p in pred
            ):
                return True
        elif isinstance(a, dict) and isinstance(pred, dict):
            if a == pred:
                return True
        elif _scalar_equal(pred, a):
            return True
    return False


def check_single_call(pred: dict, ref: dict, schema: dict | None) -> CallResult:
    """Check one predicted call against one reference call (+ optional schema).

    ``pred`` / ``ref`` shape: ``{"name": str, "arguments": {param: value(s)}}``.
    For ``ref``, each argument value is a *list of acceptable values*.
    ``schema`` is the OpenAI tool schema for the function (for required/extra checks).
    """
    if (pred.get("name") or "") != (ref.get("name") or ""):
        return CallResult(False, f"wrong function: {pred.get('name')!r} != {ref.get('name')!r}")

    pred_args = pred.get("arguments") or {}
    ref_args = ref.get("arguments") or {}

    props: dict = {}
    required: list[str] = []
    if schema:
        params = (schema.get("function") or {}).get("parameters") or {}
        props = params.get("properties") or {}
        required = params.get("required") or []

    # 4. no hallucinated parameters (only enforce when we have a schema)
    if props:
        for p in pred_args:
            if p not in props:
                return CallResult(False, f"unknown parameter {p!r}")

    # 2. required parameters present
    for p in required:
        if p not in pred_args:
            return CallResult(False, f"missing required parameter {p!r}")

    # 3. every reference argument matches
    for p, acceptable in ref_args.items():
        if p not in pred_args:
            acc = acceptable if isinstance(acceptable, list) else [acceptable]
            if any(s in acc for s in _OPTIONAL_SENTINELS):
                continue  # omittable optional argument
            return CallResult(False, f"missing parameter {p!r}")
        if not _value_matches(pred_args[p], acceptable):
            return CallResult(False, f"bad value for {p!r}: {pred_args[p]!r}")

    return CallResult(True, "ok")


def _schema_for(name: str, tools: list[dict] | None) -> dict | None:
    for t in tools or []:
        if (t.get("function") or {}).get("name") == name:
            return t
    return None


def check_ast(pred_calls: list[dict], ref_calls: list[dict], tools: list[dict] | None) -> CallResult:
    """Score a (possibly multi-call) prediction against the reference call list.

    Single-call categories: ``ref_calls`` has length 1.
    Parallel / multiple categories: each reference call must be matched by a
    distinct predicted call, and counts must be equal.
    """
    if not ref_calls:
        return CallResult(False, "no reference calls")
    if len(pred_calls) != len(ref_calls):
        return CallResult(
            False, f"call count {len(pred_calls)} != expected {len(ref_calls)}"
        )

    used: set[int] = set()
    for ref in ref_calls:
        schema = _schema_for(ref.get("name", ""), tools)
        matched = False
        for i, pred in enumerate(pred_calls):
            if i in used:
                continue
            if check_single_call(pred, ref, schema).correct:
                used.add(i)
                matched = True
                break
        if not matched:
            return CallResult(False, f"no predicted call matched reference {ref.get('name')!r}")
    return CallResult(True, "ok")
