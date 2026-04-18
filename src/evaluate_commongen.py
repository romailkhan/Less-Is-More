"""CommonGen evaluation: concept coverage + optional BLEU vs HF references."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from Cortex import Cortex
from CortexSingle import CortexSingle
from agents.Reasoning_benchmark import TASK_COMMONGEN
from dataset_loaders import Source, load_commongen_rows
from eval_answer import (
    all_concepts_used,
    concept_coverage_score,
    extract_commongen_answer_multi,
    extract_commongen_answer_single,
    max_sentence_bleu_vs_refs,
)
from retry_util import retry_on_connection_error


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_commongen_eval(
    mode: str,
    data_path: Optional[Path] = None,
    data_source: Source = "auto",
    prefer_local_hard: bool = True,
    limit: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    data = load_commongen_rows(
        source=data_source,
        jsonl_path=data_path,
        prefer_local_hard=prefer_local_hard,
    )
    if limit is not None:
        data = data[:limit]

    enable_ref = mode == "refined"
    use_multi = mode in ("multi", "refined")

    cs: Optional[CortexSingle] = None
    cm: Optional[Cortex] = None
    if use_multi:
        cm = Cortex()
    else:
        rr = "monolith" if mode == "monolith" else "single"
        cs = CortexSingle(reasoning_role=rr)

    coverages: List[float] = []
    strict_hits = 0
    bleu_scores: List[float] = []
    rows_out: List[Dict[str, Any]] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for i, row in enumerate(data, 1):
        concepts = row["concepts"]
        references = row.get("references") or []
        query = (
            "Generate a coherent sentence using all these concepts: "
            + ", ".join(concepts)
            + "."
        )

        if use_multi:
            assert cm is not None
            state = retry_on_connection_error(
                cm.process_query,
                query,
                topic="CommonGen",
                task_type=TASK_COMMONGEN,
                enable_refinement=enable_ref,
                max_refinement_iterations=3,
            )
            sentence = extract_commongen_answer_multi(state) or ""
        else:
            assert cs is not None
            state = retry_on_connection_error(
                cs.process_query,
                query,
                topic="CommonGen",
                task_type=TASK_COMMONGEN,
            )
            sentence = extract_commongen_answer_single(state) or ""

        cov = concept_coverage_score(concepts, sentence)
        coverages.append(cov)
        if all_concepts_used(concepts, sentence):
            strict_hits += 1

        bleu = max_sentence_bleu_vs_refs(sentence, references)
        if bleu is not None:
            bleu_scores.append(bleu)

        tu = state.get("token_usage") if isinstance(state, dict) else None
        if isinstance(tu, dict):
            for k in usage_total:
                usage_total[k] += int(tu.get(k) or 0)

        rows_out.append(
            {
                "index": i,
                "concepts": concepts,
                "sentence": sentence,
                "concept_coverage": cov,
                "all_concepts": all_concepts_used(concepts, sentence),
                "max_bleu_vs_ref": bleu,
                "token_usage": tu,
            }
        )
        print(
            f"[{mode}] {i}/{len(data)} coverage={cov:.3f} all={all_concepts_used(concepts, sentence)}"
        )

    n = len(data)
    avg_cov = sum(coverages) / n if n else 0.0
    strict_acc = strict_hits / n if n else 0.0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None

    summary = {
        "benchmark": "commongen",
        "mode": mode,
        "data_source": data_source,
        "prefer_local_hard": prefer_local_hard,
        "average_concept_coverage": avg_cov,
        "strict_all_concepts_rate": strict_acc,
        "average_max_bleu_vs_references": avg_bleu,
        "bleu_evaluated_count": len(bleu_scores),
        "total": n,
        "token_usage_total": usage_total,
        "rows": rows_out,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out_path}")

    print(f"\n--- CommonGen {mode} ---")
    print(f"Avg concept coverage: {avg_cov:.2%}")
    print(f"All-concepts strict: {strict_acc:.2%} ({strict_hits}/{n})")
    if avg_bleu is not None:
        print(f"Avg max BLEU vs refs: {avg_bleu:.4f} (n={len(bleu_scores)})")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=("monolith", "single", "multi", "refined"),
        default="single",
    )
    p.add_argument("--data", type=Path, default=None)
    p.add_argument(
        "--data-source",
        choices=("auto", "huggingface", "jsonl"),
        default="auto",
        help="CommonGen: local commongen_hard.jsonl, HF allenai/common_gen, or auto",
    )
    p.add_argument(
        "--commongen-hub",
        action="store_true",
        help="In auto mode, use Hugging Face test split instead of local commongen_hard.jsonl",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    root = _project_root()
    out = args.out or (root / "evaluation_results" / f"commongen_{args.mode}.json")
    os.chdir(root)
    run_commongen_eval(
        args.mode,
        args.data,
        args.data_source,
        prefer_local_hard=not args.commongen_hub,
        limit=args.limit,
        out_path=out,
    )


if __name__ == "__main__":
    main()
