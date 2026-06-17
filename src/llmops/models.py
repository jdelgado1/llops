"""Thin helpers to call model deployments via the Foundry project's OpenAI client.

Used for the *frozen-context* paths (baseline eval + judging), where we hand the
model a fixed context and ask it to answer/grade — no live web search. Live
grounded answering lives in ``teacher.py``.
"""
from __future__ import annotations

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from .config import Settings

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

    ``model`` is the deployment name (e.g. ``gpt-5.4`` or ``gpt-4.1-mini``).
    """
    prompt = (
        f"{ANSWER_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text


def answer_closed_book(client, model: str, question: str) -> str:
    """Answer with no context and no web search (parametric knowledge only).

    Used to show the 'floor' — why grounding is needed at all.
    """
    prompt = f"{CLOSED_BOOK_INSTRUCTIONS}\n\nQuestion: {question}"
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text
