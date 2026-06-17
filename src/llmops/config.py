"""Central configuration, loaded from environment / .env.

Keep this boring and explicit — an infra/data audience should be able to read
it and know exactly what endpoints and models are in play.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for the distillation loop."""

    project_endpoint: str
    teacher_model: str
    student_model: str
    judge_model: str
    student_finetuned_deployment: str | None
    websearch_country: str
    websearch_city: str
    websearch_region: str


def get_settings() -> Settings:
    """Build Settings from the environment, failing fast on the one required value."""
    endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "").strip()
    if not endpoint:
        raise RuntimeError(
            "FOUNDRY_PROJECT_ENDPOINT is not set. Copy .env.example to .env and fill it in."
        )
    return Settings(
        project_endpoint=endpoint,
        teacher_model=os.environ.get("TEACHER_MODEL", "gpt-5.4"),
        student_model=os.environ.get("STUDENT_MODEL", "gpt-4.1-mini"),
        judge_model=os.environ.get("JUDGE_MODEL", "gpt-5.4"),
        student_finetuned_deployment=os.environ.get("STUDENT_FINETUNED_DEPLOYMENT") or None,
        websearch_country=os.environ.get("WEBSEARCH_COUNTRY", "US"),
        websearch_city=os.environ.get("WEBSEARCH_CITY", "New York"),
        websearch_region=os.environ.get("WEBSEARCH_REGION", "NY"),
    )
