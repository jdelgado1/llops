"""The single headline metric: the Answer Quality Score.

We keep it deliberately simple (one number, legible to an infra/data audience):

    Answer Quality Score = % of evaluated questions an LLM judge marks CORRECT,
    where "correct" means the candidate answer conveys the same key fact(s) as
    the reference answer.

The judge is a strong model (GPT-5.4 by default). For the frozen-context
regression eval the candidate also had to work from the provided context, so a
correct answer is implicitly a grounded one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

JUDGE_INSTRUCTIONS = (
    "You are a strict grader. Decide whether the CANDIDATE answer is correct for "
    "the QUESTION, using the REFERENCE answer(s) as ground truth. Mark "
    "correct=true only if the candidate conveys the same key fact(s) as a "
    "reference answer (minor wording or formatting differences are fine). "
    "If the candidate is wrong, evasive, or omits the key fact, mark "
    "correct=false. Respond with ONLY a JSON object of the form "
    '{"correct": true, "reason": "<one short sentence>"}.'
)


@dataclass
class Judgment:
    correct: bool
    reason: str


def judge_answer(client, judge_model: str, question: str, references: list[str], candidate: str) -> Judgment:
    """Grade one candidate answer against the reference answer(s)."""
    refs = "\n".join(f"- {r}" for r in references) if references else "(none provided)"
    prompt = (
        f"{JUDGE_INSTRUCTIONS}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER(S):\n{refs}\n\n"
        f"CANDIDATE ANSWER:\n{candidate}"
    )
    resp = client.responses.create(model=judge_model, input=prompt)
    return _parse(resp.output_text)


def _parse(text: str) -> Judgment:
    """Pull a {correct, reason} object out of the judge's reply, defensively."""
    raw = (text or "").strip()
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        obj = json.loads(raw[start:end])
        return Judgment(correct=bool(obj.get("correct", False)), reason=str(obj.get("reason", "")))
    except (ValueError, json.JSONDecodeError):
        return Judgment(correct=False, reason=f"parse_error: {raw[:120]}")
