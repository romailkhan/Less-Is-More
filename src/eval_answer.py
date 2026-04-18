"""Helpers to extract model answers from Cortex / CortexSingle state dicts."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    s = str(val).strip()
    # strip common wrappers
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_mgsm_answer_single(state: Dict[str, Any]) -> Optional[float]:
    """CortexSingle: agents.reasoning.final_answer"""
    try:
        fa = state["agents"]["reasoning"]["final_answer"]
        return _to_float(fa)
    except (KeyError, TypeError):
        return None


def extract_mgsm_answer_multi(state: Dict[str, Any]) -> Optional[float]:
    """Cortex: agents.language.analysis.final_response"""
    try:
        fr = state["agents"]["language"]["analysis"]["final_response"]
        return _to_float(fr)
    except (KeyError, TypeError):
        return None


def extract_logic_answer_single(state: Dict[str, Any]) -> Optional[str]:
    try:
        fa = state["agents"]["reasoning"]["final_answer"]
        if fa is None:
            return None
        return str(fa).strip()
    except (KeyError, TypeError):
        return None


def extract_logic_answer_multi(state: Dict[str, Any]) -> Optional[str]:
    try:
        fr = state["agents"]["language"]["analysis"]["final_response"]
        if fr is None:
            return None
        return str(fr).strip()
    except (KeyError, TypeError):
        return None


def extract_commongen_answer_single(state: Dict[str, Any]) -> Optional[str]:
    try:
        fa = state["agents"]["reasoning"]["final_answer"]
        if fa is None:
            return None
        return str(fa).strip()
    except (KeyError, TypeError):
        return None


def extract_commongen_answer_multi(state: Dict[str, Any]) -> Optional[str]:
    try:
        fr = state["agents"]["language"]["analysis"]["final_response"]
        if fr is None:
            return None
        return str(fr).strip()
    except (KeyError, TypeError):
        return None


def concept_coverage_score(concepts: list[str], sentence: str) -> float:
    """Fraction of concepts that appear as substrings in the sentence (case-insensitive)."""
    if not concepts:
        return 0.0
    s = sentence.lower()
    hits = sum(1 for c in concepts if c.lower() in s)
    return hits / len(concepts)


def all_concepts_used(concepts: list[str], sentence: str) -> bool:
    s = sentence.lower()
    return all(c.lower() in s for c in concepts)


def max_sentence_bleu_vs_refs(prediction: str, references: List[str]) -> Optional[float]:
    """
    Sentence BLEU vs reference set (0..1). Requires ``sacrebleu``; returns None if unavailable.
    """
    refs = [r for r in references if r and r.strip()]
    if not refs:
        return None
    try:
        import sacrebleu
    except ImportError:
        return None
    try:
        s = sacrebleu.sentence_bleu(prediction, refs)
        return float(s.score) / 100.0
    except Exception:
        return None
