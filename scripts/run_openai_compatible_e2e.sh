#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the OpenAI-compatible endpoint}"
MODEL="${MODEL:-gpt-4o-mini}"

python -m rlm_gepa_retrieval generate --out generated/openai_compatible_tasks.jsonl --n 18 --seed 11
python -m rlm_gepa_retrieval eval \
  --corpus examples/demo_corpus \
  --questions generated/openai_compatible_tasks.jsonl \
  --program heuristic \
  --out runs/heuristic-eval.json
python -m rlm_gepa_retrieval eval \
  --corpus examples/demo_corpus \
  --questions generated/openai_compatible_tasks.jsonl \
  --program single_shot_rag \
  --out runs/single-shot-rag-eval.json
python -m rlm_gepa_retrieval eval \
  --corpus examples/demo_corpus \
  --questions generated/openai_compatible_tasks.jsonl \
  --program rlm \
  --model "$MODEL" \
  --out runs/rlm-eval.json
python -m rlm_gepa_retrieval optimize-gepa \
  --data generated/openai_compatible_tasks.jsonl \
  --model "$MODEL" \
  --reflection-model "$MODEL" \
  --artifact artifacts/optimized-rlm.json \
  --log-dir runs/gepa \
  --max-metric-calls 12
python -m rlm_gepa_retrieval eval \
  --corpus examples/demo_corpus \
  --questions generated/openai_compatible_tasks.jsonl \
  --program optimized \
  --model "$MODEL" \
  --artifact artifacts/optimized-rlm.json \
  --out runs/optimized-eval.json
python -m rlm_gepa_retrieval report \
  --runs runs/heuristic-eval.json runs/single-shot-rag-eval.json runs/rlm-eval.json runs/optimized-eval.json \
  --out reports/openai-compatible-e2e.md
