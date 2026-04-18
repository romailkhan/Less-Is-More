"""Render a markdown table for paper-style results."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set


def _pct(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _bleu(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "—"


def _mode_order() -> List[str]:
    return ["monolith", "single", "multi", "refined"]


def _mode_label(mode: str) -> str:
    return {
        "monolith": "Monolith (1× LLM)",
        "single": "CORTEXS",
        "multi": "CORTEXM",
        "refined": "CORTEXMR",
    }.get(mode, mode)


def _collect_modes(table: Dict[str, Dict[str, Any]]) -> List[str]:
    seen: Set[str] = set()
    for bench in table.values():
        if isinstance(bench, dict):
            seen.update(bench.keys())
    order = _mode_order()
    return [m for m in order if m in seen] + sorted(seen.difference(order))


def render_paper_table_md(table: Dict[str, Dict[str, Any]]) -> str:
    """
    ``table`` is ``{ "mgsm": {mode: metrics}, "commongen": {...}, "logic": {...} }``.
    """
    modes = _collect_modes(table)
    lines = [
        "# Results table",
        "",
        "Modes: **Monolith** = one LLM call per item (`MODEL_MONOLITH`); **CORTEXS** = single-agent; "
        "**CORTEXM** = multi-agent pipeline; **CORTEXMR** = multi-agent + refinement.",
        "",
        "| Architecture | CommonGen avg coverage | CommonGen all-concepts | CommonGen avg BLEU* | MGSM accuracy | Logic accuracy |",
        "|----------------|------------------------:|------------------------:|--------------------:|--------------:|---------------:|",
    ]
    for mode in modes:
        c = table.get("commongen", {}).get(mode) or {}
        m = table.get("mgsm", {}).get(mode) or {}
        lg = table.get("logic", {}).get(mode) or {}
        lines.append(
            "| {label} | {cov} | {strict} | {bleu} | {mgsm} | {logic} |".format(
                label=_mode_label(mode),
                cov=_pct(c.get("average_concept_coverage")),
                strict=_pct(c.get("strict_all_concepts_rate")),
                bleu=_bleu(c.get("average_max_bleu_vs_references")),
                mgsm=_pct(m.get("accuracy")),
                logic=_pct(lg.get("accuracy")),
            )
        )
    lines.extend(
        [
            "",
            "*BLEU* column uses reference sentences when available (Hugging Face CommonGen split); "
            "install `sacrebleu`. Local `commongen_hard.jsonl` has no references, so BLEU shows as —.",
            "",
        ]
    )
    return "\n".join(lines)


def write_paper_table_md(table: Dict[str, Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_paper_table_md(table), encoding="utf-8")
