#!/usr/bin/env python3
"""
Run benchmark matrix: MGSM, CommonGen, Logic × selected modes.

  Default modes: single, multi, refined. Add ``monolith`` for one-call-per-item baseline
  (``MODEL_MONOLITH`` vs orchestration ``MODEL_ORCHESTRATION``).

Examples::

  cd src && PYTHONPATH=. python run_all_experiments.py --limit 5
  cd src && PYTHONPATH=. python run_all_experiments.py --modes monolith,single,multi,refined
  cd src && PYTHONPATH=. python run_all_experiments.py --data-source huggingface
  cd src && PYTHONPATH=. python run_all_experiments.py --commongen-hub
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_commongen import run_commongen_eval  # noqa: E402
from evaluate_logic import run_logic_eval  # noqa: E402
from evaluate_mgsm import run_mgsm_eval  # noqa: E402
from paper_table import write_paper_table_md  # noqa: E402


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmark evaluation matrix")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items per benchmark (default: full datasets)",
    )
    parser.add_argument(
        "--only",
        choices=("mgsm", "commongen", "logic", "all"),
        default="all",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Benchmark names to skip",
    )
    parser.add_argument(
        "--data-source",
        choices=("auto", "huggingface", "jsonl"),
        default="auto",
        help="MGSM/CommonGen loading: HF hub, local jsonl, or auto (see dataset_loaders)",
    )
    parser.add_argument(
        "--commongen-hub",
        action="store_true",
        help="Use Hugging Face CommonGen test split instead of local commongen_hard.jsonl (when data-source allows)",
    )
    parser.add_argument(
        "--mgsm-offset",
        type=int,
        default=0,
        help="Skip first N MGSM problems (resume after hang), only affects MGSM runs",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="single,multi,refined",
        help="Comma-separated: monolith,single,multi,refined (default: single,multi,refined)",
    )
    args = parser.parse_args()

    root = _root()
    os.chdir(root)
    out_dir = root / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    modes: List[str] = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid = {"monolith", "single", "multi", "refined"}
    bad = [m for m in modes if m not in valid]
    if bad:
        print(f"Invalid --modes entries: {bad}; allowed: {sorted(valid)}", file=sys.stderr)
        sys.exit(1)
    table: Dict[str, Dict[str, Any]] = {}

    ds = args.data_source
    prefer_local = not args.commongen_hub

    benchmarks = ["mgsm", "commongen", "logic"]
    if args.only != "all":
        benchmarks = [args.only]

    for bench in benchmarks:
        if bench in args.skip:
            continue
        table[bench] = {}
        for mode in modes:
            print(f"\n========== {bench} / {mode} ==========")
            if bench == "mgsm":
                path = out_dir / f"mgsm_{mode}.json"
                r = run_mgsm_eval(
                    mode,
                    data_path=None,
                    data_source=ds,
                    offset=args.mgsm_offset,
                    limit=args.limit,
                    out_path=path,
                )
                table[bench][mode] = {
                    "accuracy": r.get("accuracy"),
                    "correct": r.get("correct"),
                    "total": r.get("total"),
                    "data_source": r.get("data_source"),
                    "token_usage_total": r.get("token_usage_total"),
                }
            elif bench == "commongen":
                path = out_dir / f"commongen_{mode}.json"
                r = run_commongen_eval(
                    mode,
                    data_path=None,
                    data_source=ds,
                    prefer_local_hard=prefer_local,
                    limit=args.limit,
                    out_path=path,
                )
                table[bench][mode] = {
                    "average_concept_coverage": r.get("average_concept_coverage"),
                    "strict_all_concepts_rate": r.get("strict_all_concepts_rate"),
                    "average_max_bleu_vs_references": r.get("average_max_bleu_vs_references"),
                    "total": r.get("total"),
                    "data_source": r.get("data_source"),
                    "prefer_local_hard": r.get("prefer_local_hard"),
                    "token_usage_total": r.get("token_usage_total"),
                }
            else:
                path = out_dir / f"logic_{mode}.json"
                r = run_logic_eval(
                    mode,
                    data_path=None,
                    data_source=ds,
                    limit=args.limit,
                    out_path=path,
                    write_per_puzzle=False,
                )
                table[bench][mode] = {
                    "accuracy": r.get("accuracy"),
                    "correct": r.get("correct"),
                    "total": r.get("total"),
                    "data_source": r.get("data_source"),
                    "token_usage_total": r.get("token_usage_total"),
                }

    master_path = out_dir / "all_experiments_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    md_path = out_dir / "paper_table.md"
    write_paper_table_md(table, md_path)

    print(f"\nMaster summary written to {master_path}")
    print(f"Paper-style table written to {md_path}")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
