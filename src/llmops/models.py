"""Thin helpers to call model deployments via the Foundry project's OpenAI client.

Used for the *frozen-context* paths (baseline eval + judging), where we hand the
model a fixed context and ask it to answer/grade — no live web search. Live
grounded answering lives in ``teacher.py``.
"""
from __future__ import annotations

import re

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from .config import Settings

# qwen3 (and other reasoning models) wrap chain-of-thought in <think>...</think>.
# Strip it so we evaluate/serve only the final answer. No-op for GPT models.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str | None) -> str:
    return _THINK_RE.sub("", text or "").strip()

ANSWER_INSTRUCTIONS = (
    "You are a TMG (Telco/Media/Gaming) research analyst. Answer the question "
    "using ONLY the provided context. Be concise and accurate. If the context "
    "does not contain the answer, say you cannot determine it from the context."
)

CLOSED_BOOK_INSTRUCTIONS = (
    "You are a TMG (Telco/Media/Gaming) research analyst. Answer the question "
    "from your own knowledge only — you have no web access. Be concise. If you "
    "do not know or cannot be sure, say you don't know."
)


def get_client(settings: Settings):
    """Return an OpenAI-compatible client bound to the Foundry project."""
    project = AIProjectClient(
        endpoint=settings.project_endpoint,
        credential=DefaultAzureCredential(),
    )
    return project.get_openai_client()


def answer_over_context(client, model: str, question: str, context: str) -> str:
    """Answer a question using only the provided (frozen) context.

    ``model`` is the deployment name (e.g. ``gpt-5.4`` or a fine-tuned qwen3-32b).
    Uses chat completions so it works for both GPT and qwen deployments.
    """
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=2048,
        messages=[
            {"role": "system", "content": ANSWER_INSTRUCTIONS},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return _strip_think(resp.choices[0].message.content)


def answer_closed_book(client, model: str, question: str) -> str:
    """Answer with no context and no web search (parametric knowledge only).

    Used to show the 'floor' — why grounding is needed at all.
    """
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=2048,
        messages=[
            {"role": "system", "content": CLOSED_BOOK_INSTRUCTIONS},
            {"role": "user", "content": question},
        ],
    )
    return _strip_think(resp.choices[0].message.content)
