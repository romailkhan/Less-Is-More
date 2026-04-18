# CORTEX / multi-agent orchestration benchmarks

Research codebase for comparing **one-call LLM baselines** vs **multi-agent orchestration** (and optional refinement) on MGSM, CommonGen, and logic-grid tasks. LLMs use **OpenRouter** (default `MODEL=z-ai/glm-5.1`). See **`RESEARCH.md`** for hypotheses, model-tier env vars, and mode definitions.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (manages Python **3.12** per `.python-version` and dependencies from `pyproject.toml` / `uv.lock`).

```bash
uv sync
```

This creates `.venv/` in the project root (gitignored). Commit **`uv.lock`** with the repo for reproducible installs. Dev tools: `uv sync --group dev`.

2. Configure `.env` in the project root:

```
OPENROUTER_API_KEY=sk-or-v1-...
MODEL=z-ai/glm-5.1

# Optional: separate models per research arm (defaults fall back to MODEL)
# MODEL_MONOLITH=...       # --mode monolith
# MODEL_SINGLE=...         # --mode single
# MODEL_ORCHESTRATION=...  # multi-agent pipeline stages
# MODEL_PERCEPTION=...     # optional per-stage overrides
```

**ChromaDB memory / embeddings:** By default the project uses Chroma’s **local** `DefaultEmbeddingFunction` (no OpenAI account). The first run may download a small ONNX model. If you previously used OpenAI embeddings, delete the folders `long_term_memory_store` and `long_term_memory_store_single_agent` so collections are recreated with the new embedding size.

Optional — use OpenAI for embeddings instead: set `CORTEX_USE_OPENAI_EMBEDDINGS=1` and `OPENAI_API_KEY=...`.

Optional: `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1` (default).

**Stuck runs:** set `OPENROUTER_TIMEOUT` (seconds, default **600**) so a single API call cannot hang indefinitely. Increase if you hit timeouts on long multi-agent chains (e.g. `OPENROUTER_TIMEOUT=1200`).

## Run a quick demo

From `src/` with `PYTHONPATH` set (`uv run` uses the project environment):

```bash
cd src
PYTHONPATH=. uv run python main.py
```

## Benchmarks

| Script | Modes | Data |
|--------|-------|------|
| `evaluate_mgsm.py` | `--mode monolith\|single\|multi\|refined` | `--data-source auto\|huggingface\|jsonl` (HF: `juletxara/mgsm` en/test) |
| `evaluate_commongen.py` | same | Default: local `commongen_hard.jsonl`; `--commongen-hub` or HF-only for `allenai/common_gen` test |
| `evaluate_logic.py` | same | Local `logic_grid_puzzle_200.jsonl` |

Run the full matrix (default modes: **single, multi, refined**):

```bash
cd src
PYTHONPATH=. uv run python run_all_experiments.py
# smoke test:
PYTHONPATH=. uv run python run_all_experiments.py --limit 2
# include one-call monolith baseline (12 runs if all benchmarks):
PYTHONPATH=. uv run python run_all_experiments.py --modes monolith,single,multi,refined
# Hugging Face MGSM + official CommonGen test (large):
PYTHONPATH=. uv run python run_all_experiments.py --data-source huggingface --commongen-hub --limit 20
```

Results are written under `evaluation_results/`: per-benchmark JSON (includes **`token_usage_total`** when the API returns usage), `all_experiments_summary.json`, and `paper_table.md`.

Set `HF_DATASETS_OFFLINE=1` to force local JSONL only (no Hub).

Hub datasets are loaded from the **`refs/convert/parquet`** branch when possible, so **`datasets` 3.x** does not need legacy Python dataset scripts (no more `trust_remote_code` / `mgsm.py` errors). If Parquet fails, MGSM/CommonGen fall back to local JSONL or, on `datasets` 2.x only, the old script loader.

## Project layout

- `RESEARCH.md` — research questions, model env vars, mode table
- `src/Cortex.py` — multi-agent pipeline; `enable_refinement=True` for refinement loops
- `src/CortexSingle.py` — single `Reasoning_Benchmark` agent; `reasoning_role` selects `MODEL_SINGLE` vs `MODEL_MONOLITH`
- `src/agents/openrouter_llm.py` — OpenRouter `ChatOpenAI`; `resolve_model_id()` for per-role models
- `src/agents/token_usage.py` — aggregates token counts per `process_query` when usage metadata is present
