"""Shared ChatOpenAI client configured for OpenRouter."""
import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI


def resolve_model_id(*, role: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Pick which OpenRouter model id to use for this client.

    Precedence:
    - explicit ``model`` argument
    - ``MODEL_<ROLE>`` (e.g. MODEL_PERCEPTION, MODEL_MONOLITH, MODEL_SINGLE)
    - ``MODEL_ORCHESTRATION`` for pipeline stages (when ``role`` is set and not single/monolith)
    - ``MODEL_SINGLE`` when role is ``single``
    - ``MODEL_MONOLITH`` when role is ``monolith``
    - ``MODEL`` (legacy default)
    """
    if model:
        return model
    default = os.getenv("MODEL", "z-ai/glm-5.1")
    if not role:
        return default
    r = role.upper()
    specific = os.getenv(f"MODEL_{r}")
    if specific:
        return specific
    if role == "single":
        return os.getenv("MODEL_SINGLE") or default
    if role == "monolith":
        return os.getenv("MODEL_MONOLITH") or default
    return os.getenv("MODEL_ORCHESTRATION") or default


def get_openrouter_llm(
    *,
    role: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointing at OpenRouter's OpenAI-compatible API.

    Use ``role`` (e.g. ``perception``, ``single``, ``monolith``) so ``MODEL_<ROLE>`` and
    ``MODEL_ORCHESTRATION`` / ``MODEL_SINGLE`` / ``MODEL_MONOLITH`` apply (see ``resolve_model_id``).

    ``OPENROUTER_TIMEOUT`` (seconds, default 600): max wait per HTTP request. Without this,
    a stuck provider connection can hang forever (see LangChain ``ChatOpenAI(timeout=...)``).
    """
    temp = temperature if temperature is not None else float(os.getenv("TEMPERATURE", "0.6"))
    mx = max_tokens if max_tokens is not None else int(os.getenv("MAX_TOKENS", "4096"))
    timeout_s = float(os.getenv("OPENROUTER_TIMEOUT", "600"))
    max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "2"))
    mid = resolve_model_id(role=role, model=model)
    return ChatOpenAI(
        model=mid,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=temp,
        max_tokens=mx,
        timeout=timeout_s,
        max_retries=max_retries,
        **kwargs,
    )
