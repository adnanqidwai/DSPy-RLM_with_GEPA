#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q
python -m rlm_gepa_retrieval demo --out runs/demo-answer.json
python -m rlm_gepa_retrieval generate --out generated/demo_tasks.jsonl --n 12 --seed 7
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
python -m rlm_gepa_retrieval report \
  --runs runs/heuristic-eval.json runs/single-shot-rag-eval.json \
  --out reports/local-smoke.md
