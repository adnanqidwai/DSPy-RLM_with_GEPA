# RLM-GEPA Retrieval

RLM-GEPA Retrieval is a DSPy RLM + GEPA harness for optimizing agentic retrieval
policies. It generates retrieval tasks over a local Markdown corpus, lets a DSPy
RLM call host-backed retrieval tools, records the tool trace, scores citations and
answer support deterministically, and uses GEPA to improve the RLM policy from
textual failure feedback.

The core loop is:

```text
generate tasks -> evaluate RLM policy -> optimize with GEPA -> evaluate optimized policy -> report
```

The repository also includes deterministic controls: a lexical heuristic and a
retrieve-once RAG baseline.

See [docs/design.md](docs/design.md) for the RLM + GEPA architecture.

## Quick Start

```bash
cd DSPy-RLM_with_GEPA
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,dspy]'
```

Configure an OpenAI-compatible chat-completions endpoint:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional for OpenAI
```

Generate a task set:

```bash
python -m rlm_gepa_retrieval generate \
  --out generated/demo_tasks.jsonl \
  --n 24 \
  --seed 7
```

The bundled task bank contains 24 split-disjoint examples. Use a smaller `--n`
for smoke runs; extend the corpus and task bank before requesting more than 24.

Evaluate the base DSPy RLM retrieval policy:

```bash
python -m rlm_gepa_retrieval eval \
  --corpus examples/demo_corpus \
  --questions generated/demo_tasks.jsonl \
  --program rlm \
  --model gpt-4o-mini \
  --out runs/rlm-eval.json
```

Optimize the RLM policy with GEPA:

```bash
python -m rlm_gepa_retrieval optimize-gepa \
  --data generated/demo_tasks.jsonl \
  --model gpt-4o-mini \
  --reflection-model gpt-4o-mini \
  --artifact artifacts/optimized-rlm.json \
  --log-dir runs/gepa \
  --max-metric-calls 12
```

Evaluate the optimized policy:

```bash
python -m rlm_gepa_retrieval eval \
  --corpus examples/demo_corpus \
  --questions generated/demo_tasks.jsonl \
  --program optimized \
  --artifact artifacts/optimized-rlm.json \
  --model gpt-4o-mini \
  --out runs/optimized-eval.json
```

Build a Markdown report:

```bash
python -m rlm_gepa_retrieval report \
  --runs runs/heuristic-eval.json runs/single-shot-rag-eval.json runs/rlm-eval.json runs/optimized-eval.json \
  --out reports/rlm-gepa-report.md
```

End-to-end script:

```bash
MODEL=gpt-4o-mini ./scripts/run_openai_compatible_e2e.sh
```

## What The RLM Does

The DSPy RLM receives a question, a compact corpus manifest, and a retrieval
budget. It can call host tools during its reasoning loop:

- `search(query, k)` for lexical passage retrieval;
- `open_doc(doc_id)` for full local document text;
- `find_in_doc(doc_id, pattern)` for targeted regex snippets.

`RetrievalEnvironment` records the actual tool calls. Evaluation uses that host
trace for budget scoring, so the model cannot get credit by merely claiming it
searched efficiently.

## What GEPA Optimizes

GEPA optimizes the textual retrieval policy inside the DSPy RLM program. The
metric returns both a score and failure feedback: missed supporting passages,
unsupported citations, missing answer terms, and wasted search budget. GEPA uses
that feedback to produce a better retrieval policy artifact.

## CLI Overview

| Command | Purpose |
| --- | --- |
| `generate` | Deterministic retrieval-task JSONL |
| `eval` | Score `heuristic`, `single_shot_rag`, `rlm`, or `optimized` programs |
| `optimize-gepa` | GEPA optimization of the DSPy RLM retrieval policy |
| `report` | Markdown report from eval JSON runs |
| `demo` | Small bundled inspection question |
| `answer` | Single-question retrieval inspection path |

## Metrics

Each eval example is scored with:

- **answer correctness** (0.35): required terms in the answer;
- **evidence recall** (0.35): gold passage IDs cited;
- **citation precision** (0.15): cited IDs that are gold support;
- **budget efficiency** (0.15): metered retrieval tool calls within `max_searches`.

## Deterministic Controls

The `heuristic` program is a lexical baseline used to validate generated tasks,
scoring, reporting, and corpus coverage. The `single_shot_rag` program performs
one top-k search and extracts answer sentences from the retrieved passages. It
is a stronger control for comparing ordinary RAG against the RLM policy.

```bash
python -m rlm_gepa_retrieval eval \
  --corpus examples/demo_corpus \
  --questions generated/demo_tasks.jsonl \
  --program heuristic \
  --out runs/heuristic-eval.json

python -m rlm_gepa_retrieval eval \
  --corpus examples/demo_corpus \
  --questions generated/demo_tasks.jsonl \
  --program single_shot_rag \
  --out runs/single-shot-rag-eval.json
```

## Development

```bash
pip install -e '.[dev,dspy]'
python -m pytest -q tests
python -m compileall -q src tests
```

The test suite covers task generation, retrieval environments, deterministic
metrics, DSPy program construction, report writing, API configuration validation,
and command-line flows.
