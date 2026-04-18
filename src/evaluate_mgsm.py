"""MGSM benchmark evaluation: single-agent, multi-agent, or multi-agent + refinement."""
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
from agents.Reasoning_benchmark import TASK_MGSM
from dataset_loaders import Source, load_mgsm_cases
from eval_answer import extract_mgsm_answer_multi, extract_mgsm_answer_single
from retry_util import retry_on_connection_error


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_mgsm_eval(
    mode: str,
    data_path: Optional[Path] = None,
    data_source: Source = "auto",
    offset: int = 0,
    limit: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    mode: monolith | single | multi | refined

    monolith: one LLM call per item; model from ``MODEL_MONOLITH`` (default ``MODEL``).
    single: one specialist agent; model from ``MODEL_SINGLE`` (default ``MODEL``).
    multi / refined: full pipeline; stages use ``MODEL_ORCHESTRATION`` / ``MODEL_<ROLE>``.

    offset: skip first N problems (resume after a crash/hang).
    """
    root = _project_root()
    all_cases = load_mgsm_cases(
        source=data_source,
        jsonl_path=data_path or (root / "src" / "data" / "mgsm" / "test.jsonl"),
    )
    n_full = len(all_cases)
    if offset:
        all_cases = all_cases[offset:]
    cases = all_cases
    if limit is not None:
        cases = cases[:limit]

    enable_ref = mode == "refined"
    use_multi = mode in ("multi", "refined")

    cortex_single: Optional[CortexSingle] = None
    cortex_multi: Optional[Cortex] = None

    if use_multi:
        cortex_multi = Cortex()
    else:
        rr = "monolith" if mode == "monolith" else "single"
        cortex_single = CortexSingle(reasoning_role=rr)

    correct = 0
    rows: List[Dict[str, Any]] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for idx, test_case in enumerate(cases):
        i = offset + idx + 1
        question = test_case["question"]
        expected = float(test_case["answer_number"])

        if use_multi:
            assert cortex_multi is not None
            state = retry_on_connection_error(
                cortex_multi.process_query,
                question,
                topic="Math Word Problem",
                task_type=TASK_MGSM,
                enable_refinement=enable_ref,
                max_refinement_iterations=3,
            )
            pred = extract_mgsm_answer_multi(state)
        else:
            assert cortex_single is not None
            state = retry_on_connection_error(
                cortex_single.process_query,
                question,
                topic="Math Word Problem",
                task_type=TASK_MGSM,
            )
            pred = extract_mgsm_answer_single(state)

        ok = pred is not None and abs(pred - expected) < 1e-5
        if ok:
            correct += 1

        tu = state.get("token_usage") if isinstance(state, dict) else None
        if isinstance(tu, dict):
            for k in usage_total:
                usage_total[k] += int(tu.get(k) or 0)

        rows.append(
            {
                "index": i,
                "expected": expected,
                "predicted": pred,
                "correct": ok,
                "token_usage": tu,
            }
        )
        print(f"[{mode}] {i}/{n_full} expected={expected} pred={pred} ok={ok}")

    total = len(cases)
    acc = correct / total if total else 0.0
    summary = {
        "benchmark": "mgsm",
        "mode": mode,
        "data_source": data_source,
        "offset": offset,
        "dataset_size": n_full,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "token_usage_total": usage_total,
        "rows": rows,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out_path}")

    print(f"\n--- MGSM {mode} ---")
    print(f"Accuracy: {acc:.2%} ({correct}/{total})")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MGSM")
    parser.add_argument(
        "--mode",
        choices=("monolith", "single", "multi", "refined"),
        default="single",
        help="monolith (1× MODEL_MONOLITH) | single (MODEL_SINGLE) | multi | refined",
    )
    parser.add_argument("--data", type=Path, default=None, help="Path to test.jsonl")
    parser.add_argument(
        "--data-source",
        choices=("auto", "huggingface", "jsonl"),
        default="auto",
        help="MGSM: Hugging Face juletxara/mgsm (en, test) or local jsonl",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N problems (e.g. 201 to resume after hang on #202)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max problems")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON summary (default: evaluation_results/mgsm_<mode>.json)",
    )
    args = parser.parse_args()
    root = _project_root()
    out = args.out or (root / "evaluation_results" / f"mgsm_{args.mode}.json")
    os.chdir(root)
    run_mgsm_eval(
        args.mode,
        args.data,
        args.data_source,
        args.offset,
        args.limit,
        out,
    )


if __name__ == "__main__":
    main()
