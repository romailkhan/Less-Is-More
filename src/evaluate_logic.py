"""Logic Grid Puzzle 200 evaluation."""
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
from agents.Reasoning_benchmark import TASK_LOGIC_GRID
from dataset_loaders import Source, load_logic_grid_rows
from eval_answer import extract_logic_answer_multi, extract_logic_answer_single
from retry_util import retry_on_connection_error


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_logic_eval(
    mode: str,
    data_path: Optional[Path] = None,
    data_source: Source = "auto",
    limit: Optional[int] = None,
    out_path: Optional[Path] = None,
    write_per_puzzle: bool = False,
) -> Dict[str, Any]:
    root = _project_root()
    puzzles = load_logic_grid_rows(
        source=data_source,
        jsonl_path=data_path or (root / "src" / "data" / "logic_grid" / "logic_grid_puzzle_200.jsonl"),
    )
    if limit is not None:
        puzzles = puzzles[:limit]

    enable_ref = mode == "refined"
    use_multi = mode in ("multi", "refined")

    cs: Optional[CortexSingle] = None
    cm: Optional[Cortex] = None
    if use_multi:
        cm = Cortex()
    else:
        rr = "monolith" if mode == "monolith" else "single"
        cs = CortexSingle(reasoning_role=rr)

    results_dir: Optional[Path] = None
    if write_per_puzzle:
        results_dir = root / "evaluation_results" / f"logic_{mode}"
        results_dir.mkdir(parents=True, exist_ok=True)

    correct_count = 0
    rows: List[Dict[str, Any]] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for puzzle in puzzles:
        puzzle_idx = puzzle.get("idx", "N/A")
        puzzle_input = puzzle.get("inputs", "")
        correct_target = puzzle.get("targets", [])

        if not puzzle_input or not correct_target:
            continue

        if use_multi:
            assert cm is not None
            raw = retry_on_connection_error(
                cm.process_query,
                puzzle_input,
                topic="Logic Grid",
                task_type=TASK_LOGIC_GRID,
                enable_refinement=enable_ref,
                max_refinement_iterations=3,
            )
            extracted = extract_logic_answer_multi(raw)
        else:
            assert cs is not None
            raw = retry_on_connection_error(
                cs.process_query,
                puzzle_input,
                topic="Logic Grid",
                task_type=TASK_LOGIC_GRID,
            )
            extracted = extract_logic_answer_single(raw)

        is_correct = extracted is not None and extracted in correct_target
        if is_correct:
            correct_count += 1

        tu = raw.get("token_usage") if isinstance(raw, dict) else None
        if isinstance(tu, dict):
            for k in usage_total:
                usage_total[k] += int(tu.get(k) or 0)

        rows.append(
            {
                "idx": puzzle_idx,
                "targets": correct_target,
                "predicted": extracted,
                "correct": is_correct,
                "token_usage": tu,
            }
        )

        if results_dir is not None:
            fp = results_dir / f"puzzle_{puzzle_idx}.txt"
            with open(fp, "w", encoding="utf-8") as f:
                f.write("=== PUZZLE INPUT ===\n")
                f.write(puzzle_input)
                f.write("\n\n=== TARGET ===\n")
                f.write(str(correct_target))
                f.write("\n\n=== PREDICTED ===\n")
                f.write(str(extracted))
                f.write("\n\n=== RAW ===\n")
                f.write(json.dumps(raw, indent=2))

        print(f"[{mode}] idx={puzzle_idx} pred={extracted} ok={is_correct}")

    total = len(rows)
    acc = correct_count / total if total else 0.0
    summary = {
        "benchmark": "logic_grid",
        "mode": mode,
        "data_source": data_source,
        "accuracy": acc,
        "correct": correct_count,
        "total": total,
        "token_usage_total": usage_total,
        "rows": rows,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out_path}")

    print(f"\n--- Logic {mode} ---")
    print(f"Accuracy: {acc:.2%} ({correct_count}/{total})")
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
        help="Logic grid always loads local JSONL; HF value is ignored (symmetry with other evals)",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--write-puzzles", action="store_true")
    args = p.parse_args()
    root = _project_root()
    out = args.out or (root / "evaluation_results" / f"logic_{args.mode}.json")
    os.chdir(root)
    run_logic_eval(
        args.mode,
        args.data,
        args.data_source,
        args.limit,
        out,
        args.write_puzzles,
    )


if __name__ == "__main__":
    main()
