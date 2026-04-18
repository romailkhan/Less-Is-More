"""Accumulate per-request token counts from LangChain AIMessages (OpenRouter / OpenAI-compatible)."""
from __future__ import annotations

from typing import Any, Dict, Optional

_active: Optional[Dict[str, int]] = None


def usage_run_begin() -> None:
    global _active
    _active = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def usage_run_end() -> Dict[str, int]:
    global _active
    out = dict(_active) if _active else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    _active = None
    return out


def extract_token_usage_from_message(msg: Any) -> Optional[Dict[str, int]]:
    """Best-effort extraction from AIMessage or raw API payload."""
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict) and um:
        inp = int(um.get("input_tokens") or um.get("prompt_tokens") or 0)
        out = int(um.get("output_tokens") or um.get("completion_tokens") or 0)
        tot = int(um.get("total_tokens") or (inp + out))
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}

    rm = getattr(msg, "response_metadata", None) or {}
    if isinstance(rm, dict):
        tu = rm.get("token_usage")
        if isinstance(tu, dict):
            inp = int(tu.get("prompt_tokens") or tu.get("input_tokens") or 0)
            out = int(tu.get("completion_tokens") or tu.get("output_tokens") or 0)
            tot = int(tu.get("total_tokens") or (inp + out))
            return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}

    return None


def _merge(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k in ("input_tokens", "output_tokens", "total_tokens"):
        dst[k] = dst.get(k, 0) + src.get(k, 0)


def record_message(msg: Any) -> None:
    global _active
    if _active is None:
        return
    u = extract_token_usage_from_message(msg)
    if u:
        _merge(_active, u)
