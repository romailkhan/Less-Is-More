# Research framing (revamped)

This codebase supports **comparing ways of using LLMs on the same benchmarks**, not only a “cognitive” story. The pipeline (perception → … → feedback) is an **orchestration pattern**: multiple model calls with structured handoffs. You can study it against simpler baselines under controlled **model assignments** and **token accounting**.

## Questions you can answer

1. **Accuracy** — Do multi-step orchestration and/or refinement improve MGSM, CommonGen, or logic-grid accuracy versus one-call baselines?
2. **Cost / compute** — For a fixed task budget, does one large monolith call beat several smaller calls? Summaries include **`token_usage_total`** when the provider returns usage metadata (OpenRouter-compatible responses).
3. **Fair model tiers** — Use env vars so “big monolith” and “small orchestra” are explicit, not accidental:

| Variable | Meaning |
|----------|---------|
| `MODEL` | Legacy default when overrides are unset |
| `MODEL_MONOLITH` | One-call baseline (`--mode monolith`) |
| `MODEL_SINGLE` | Single-agent specialist (`--mode single`) |
| `MODEL_ORCHESTRATION` | Default for every stage of the multi-agent pipeline |
| `MODEL_PERCEPTION`, `MODEL_EMOTION`, `MODEL_REASONING`, `MODEL_LANGUAGE`, `MODEL_FEEDBACK` | Per-stage overrides |

Example: `MODEL_MONOLITH=.../large` and `MODEL_ORCHESTRATION=.../small` to test *one strong model* vs *many weak calls*.

## Modes (eval scripts / `run_all_experiments.py`)

| Mode | What runs |
|------|-----------|
| `monolith` | One `Reasoning_Benchmark` call per item (`MODEL_MONOLITH`) |
| `single` | Same architecture as monolith, but **`MODEL_SINGLE`** — use when the baseline is “specialist agent” with a different model id than monolith |
| `multi` | Full pipeline (`MODEL_ORCHESTRATION` / per-role) |
| `refined` | Multi-agent + feedback-driven refinement loop |

`monolith` and `single` share the same code path; they differ only by which env key selects the model. Set both to the same model id for identical behavior.

## Artifacts

- Per-run JSON under `evaluation_results/` includes **`token_usage_total`** (and per-row `token_usage` when present).
- `paper_table.md` lists whichever modes you included in the last matrix (`--modes` in `run_all_experiments.py`).

Cognitive labels in prompts are **narrative**; the empirical core is **architecture × model assignment × metrics**.
