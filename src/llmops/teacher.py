"""The grounded teacher: a Foundry Prompt Agent backed by the Web Search tool.

Grounding is powered by Microsoft Web IQ via the GA **Web Search tool** in the
Microsoft Foundry Agents API. We use this (not the deprecated classic
"Grounding with Bing Search") because it needs no extra Azure resource and it
supports GPT-5-class models.

Two ways to ask:
  - ``ask(question)``                  -> live web search (used for trace
                                          generation and the drift eval slice)
  - ``answer_over_context(q, ctx)``    -> synthesize over *provided* frozen
                                          context (used for the reproducible
                                          regression eval; no live search)

Usage::

    with GroundedTeacher(get_settings()) as teacher:
        ans = teacher.ask("What shipped in 5G FWA this week?")
        print(ans.answer, ans.citations)
"""
from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass, field

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    WebSearchApproximateLocation,
    WebSearchTool,
)
from azure.identity import DefaultAzureCredential

from .config import Settings

DEFAULT_INSTRUCTIONS = (
    "You are a TMG (Telco/Media/Gaming) market & competitive-intelligence "
    "research analyst. Answer the question using web search for current, "
    "factual information. Be concise and accurate, and cite your sources. "
    "If the question rests on a false premise, say so."
)

CONTEXT_INSTRUCTIONS = (
    "You are a TMG research analyst. Answer the question using ONLY the "
    "provided context. Be concise and accurate. If the context does not "
    "contain the answer, say you cannot determine it from the context."
)


@dataclass
class GroundedAnswer:
    """A teacher answer plus the source URLs it cited."""

    answer: str
    citations: list[str] = field(default_factory=list)


def _extract_citations(response) -> list[str]:
    """Pull url_citation annotations out of a Responses API result, defensively."""
    urls: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            for ann in getattr(content, "annotations", None) or []:
                url = getattr(ann, "url", None)
                if getattr(ann, "type", None) == "url_citation" and url:
                    urls.append(url)
    # de-dupe, preserve order
    seen: set[str] = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


class GroundedTeacher(contextlib.AbstractContextManager):
    """Creates a Prompt Agent version with the Web Search tool and reuses it.

    ``model`` defaults to the configured teacher, but can be overridden (e.g. to
    run the *student* deployment grounded for an apples-to-apples web_search eval).
    """

    def __init__(self, settings: Settings, instructions: str = DEFAULT_INSTRUCTIONS, model: str | None = None):
        self._settings = settings
        self._instructions = instructions
        self._model = model or settings.teacher_model
        self._project: AIProjectClient | None = None
        self._openai = None
        self._agent = None

    def __enter__(self) -> "GroundedTeacher":
        s = self._settings
        self._project = AIProjectClient(
            endpoint=s.project_endpoint,
            credential=DefaultAzureCredential(),
        )
        self._openai = self._project.get_openai_client()
        self._agent = self._project.agents.create_version(
            agent_name=f"tmg-grounded-{uuid.uuid4().hex[:8]}",
            definition=PromptAgentDefinition(
                model=self._model,
                instructions=self._instructions,
                tools=[
                    WebSearchTool(
                        user_location=WebSearchApproximateLocation(
                            country=s.websearch_country,
                            city=s.websearch_city,
                            region=s.websearch_region,
                        )
                    )
                ],
            ),
            description="TMG grounded research teacher (Web Search / WebIQ).",
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._project and self._agent is not None:
            with contextlib.suppress(Exception):
                self._project.agents.delete_version(
                    agent_name=self._agent.name, agent_version=self._agent.version
                )

    def ask(self, question: str) -> GroundedAnswer:
        """Answer with LIVE web search (forces the tool so we always ground)."""
        resp = self._openai.responses.create(
            tool_choice="required",
            input=question,
            extra_body={
                "agent_reference": {"name": self._agent.name, "type": "agent_reference"}
            },
        )
        return GroundedAnswer(answer=resp.output_text, citations=_extract_citations(resp))

    def answer_over_context(self, question: str, context: str) -> GroundedAnswer:
        """Answer using ONLY provided frozen context (no live search).

        Used for the reproducible regression eval, where the retrieved context
        is fixed so scores are stable run-to-run.
        """
        prompt = (
            f"{CONTEXT_INSTRUCTIONS}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        resp = self._openai.responses.create(
            model=self._model,
            input=prompt,
        )
        return GroundedAnswer(answer=resp.output_text, citations=[])
