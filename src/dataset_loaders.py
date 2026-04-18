"""
Load benchmark data from Hugging Face Datasets (preferred) or local JSONL files.

Environment:
  HF_DATASETS_OFFLINE=1  — skip Hub, use jsonl only
  HF_HOME / cache — standard Hugging Face cache dirs

Hub loads use the ``refs/convert/parquet`` snapshot when possible so ``datasets`` 3.x works without legacy Python loading scripts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

Source = Literal["auto", "huggingface", "jsonl"]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _hf_available() -> bool:
    if os.getenv("HF_DATASETS_OFFLINE", "").lower() in ("1", "true", "yes"):
        return False
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


def _datasets_major_version() -> int:
    try:
        import datasets

        return int(datasets.__version__.split(".")[0])
    except Exception:
        return 0


def _load_mgsm_from_hub(
    hf_dataset: str,
    hf_config: str,
    hf_split: str,
):
    """
    Prefer Parquet snapshot (``refs/convert/parquet``) — works with ``datasets`` 3.x, no ``mgsm.py`` script.

    Fall back to script + ``trust_remote_code`` only on ``datasets`` 2.x.
    """
    from datasets import load_dataset

    try:
        return load_dataset(
            hf_dataset,
            hf_config,
            split=hf_split,
            revision="refs/convert/parquet",
        )
    except Exception:
        pass

    if _datasets_major_version() < 3:
        return load_dataset(
            hf_dataset,
            hf_config,
            split=hf_split,
            trust_remote_code=True,
        )
    raise RuntimeError(
        "MGSM Parquet snapshot failed and dataset scripts are unsupported on datasets>=3. "
        "Use local src/data/mgsm/test.jsonl or pin datasets<3."
    )


def _load_commongen_from_hub(
    hf_dataset: str,
    hf_split: str,
):
    """Parquet branch first (datasets 3+); legacy script only on datasets 2.x."""
    from datasets import load_dataset

    for loader in (
        lambda: load_dataset(
            hf_dataset,
            split=hf_split,
            revision="refs/convert/parquet",
        ),
        lambda: load_dataset(
            hf_dataset,
            "default",
            split=hf_split,
            revision="refs/convert/parquet",
        ),
    ):
        try:
            return loader()
        except Exception:
            continue

    if _datasets_major_version() < 3:
        return load_dataset(
            hf_dataset,
            split=hf_split,
            trust_remote_code=True,
        )
    raise RuntimeError(
        "CommonGen Parquet snapshot failed; dataset scripts require datasets<3. "
        "Use local commongen_hard.jsonl or pin datasets<3."
    )


def load_mgsm_cases(
    *,
    source: Source = "auto",
    jsonl_path: Optional[Path] = None,
    hf_dataset: str = "juletxara/mgsm",
    hf_config: str = "en",
    hf_split: str = "test",
) -> List[Dict[str, Any]]:
    """
    MGSM items: question, answer_number (float-compatible).

    Hub default: juletxara/mgsm config ``en``, split ``test`` (250 rows).
    """
    local = jsonl_path or (_project_root() / "src" / "data" / "mgsm" / "test.jsonl")

    if source == "jsonl":
        if not local.is_file():
            raise FileNotFoundError(f"MGSM jsonl not found: {local}")
        cases_jsonl: List[Dict[str, Any]] = []
        with open(local, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cases_jsonl.append(
                    {
                        "question": obj["question"],
                        "answer_number": float(obj["answer_number"]),
                    }
                )
        return cases_jsonl

    use_hf = source == "huggingface" or (source == "auto" and _hf_available())
    if use_hf:
        try:
            ds = _load_mgsm_from_hub(hf_dataset, hf_config, hf_split)
            out: List[Dict[str, Any]] = []
            for row in ds:
                out.append(
                    {
                        "question": row["question"],
                        "answer_number": float(row["answer_number"]),
                    }
                )
            return out
        except Exception as e:
            if source == "huggingface":
                raise RuntimeError(
                    f"Failed to load MGSM from Hugging Face ({hf_dataset}/{hf_config}): {e}"
                ) from e
            print(
                f"MGSM: Hub load failed ({e}); using local {local.name} (same test split)."
            )

    if not local.is_file():
        raise FileNotFoundError(f"MGSM jsonl not found: {local}")
    cases: List[Dict[str, Any]] = []
    with open(local, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                {
                    "question": obj["question"],
                    "answer_number": float(obj["answer_number"]),
                }
            )
    return cases


def load_commongen_rows(
    *,
    source: Source = "auto",
    jsonl_path: Optional[Path] = None,
    hf_dataset: str = "allenai/common_gen",
    hf_split: str = "test",
    dedupe_concepts: bool = True,
    prefer_local_hard: bool = True,
) -> List[Dict[str, Any]]:
    """
    CommonGen rows: concepts (list[str]), optional references (list of human refs).

    Hub: allenai/common_gen — test split; dedupe by concept set (multiple refs per set).
    Local: commongen_hard.jsonl (one concept list per line).

    **auto:** if ``commongen_hard.jsonl`` exists and ``prefer_local_hard``, load it (paper subset); else Hugging Face.

    Set ``prefer_local_hard=False`` to use the Hub in **auto** mode (e.g. full official test split).
    """
    local = jsonl_path or (
        _project_root() / "src" / "data" / "commongen" / "commongen_hard.jsonl"
    )

    if source == "jsonl":
        if not local.is_file():
            raise FileNotFoundError(f"CommonGen jsonl not found: {local}")
        rows_loc: List[Dict[str, Any]] = []
        with open(local, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows_loc.append({"concepts": obj["concepts"], "references": []})
        return rows_loc

    if source == "auto" and prefer_local_hard and local.is_file():
        rows: List[Dict[str, Any]] = []
        with open(local, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append({"concepts": obj["concepts"], "references": []})
        return rows

    use_hf = source == "huggingface" or (source == "auto" and _hf_available())
    if use_hf:
        try:
            from collections import defaultdict

            ds = _load_commongen_from_hub(hf_dataset, hf_split)
            if dedupe_concepts:
                grouped: Dict[tuple, List[str]] = defaultdict(list)
                for row in ds:
                    key = tuple(row["concepts"])
                    t = (row.get("target") or "").strip()
                    if t:
                        grouped[key].append(t)
                rows = [
                    {
                        "concepts": list(k),
                        "references": list(dict.fromkeys(v))[:20],
                    }
                    for k, v in sorted(grouped.items(), key=lambda x: x[0])
                ]
            else:
                rows = [
                    {
                        "concepts": list(row["concepts"]),
                        "references": [(row.get("target") or "").strip()],
                    }
                    for row in ds
                ]
            return rows
        except Exception as e:
            if source == "huggingface":
                raise RuntimeError(
                    f"Failed to load CommonGen from Hugging Face ({hf_dataset}): {e}"
                ) from e
            print(f"Warning: CommonGen Hugging Face load failed ({e}); falling back to {local}")

    if not local.is_file():
        raise FileNotFoundError(f"CommonGen jsonl not found: {local}")
    rows = []
    with open(local, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"concepts": obj["concepts"], "references": []})
    return rows


def load_logic_grid_rows(
    *,
    source: Source = "auto",
    jsonl_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Logic grid puzzles: idx, inputs, targets (list of str).

    Not on a standard HF hub — always local JSONL (paper's Logic Grid Puzzle 200).
    """
    _ = source  # reserved if a Hub dataset appears later
    local = jsonl_path or (
        _project_root() / "src" / "data" / "logic_grid" / "logic_grid_puzzle_200.jsonl"
    )
    if not local.is_file():
        raise FileNotFoundError(f"Logic grid jsonl not found: {local}")
    puzzles: List[Dict[str, Any]] = []
    with open(local, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            puzzles.append(json.loads(line))
    return puzzles
