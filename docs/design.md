# RLM-GEPA Retrieval Design

## Objective

Build a standalone retrieval-policy evaluation harness where a DSPy RLM explores a local Markdown
corpus through host-backed retrieval tools under a tool-call budget, emits citations, and GEPA
optimizes the retrieval policy from deterministic textual feedback.

Deterministic controls remain available for validating task generation, metrics,
reports, and corpus coverage. The main research path is the DSPy RLM plus GEPA optimizer.

## Non-Goals

- No model training or weight updates.
- No claim of state-of-the-art benchmark performance.
- No mandatory vector database.
- No hidden web dependency in the default path.
- No LLM-as-judge requirement for the core eval.

## Task Families

Generated retrieval examples use these families:

- `single_hop`
- `multi_hop_bridge`
- `conflict_resolution`
- `acronym_alias`
- `needle_section`
- `aggregation`

Each example specifies `question`, `expected_answer`, `required_terms`, `required_passage_ids`,
and `max_searches`.

The bundled bank contains 24 examples: 8 train, 8 dev, and 8 test examples with
split-disjoint gold passage IDs. The generator rejects requests above that bank
size instead of cycling duplicate questions.

## Retrieval Controller

The DSPy RLM program uses:

```text
question, corpus_json, tool_budget -> answer, citation_ids, trace
```

`corpus_json` is a compact corpus manifest. The RLM gets full evidence only by calling host-backed
tools registered through DSPy's `tools=` interface:

- `search(query, k)` — lexical BM25-style retrieval
- `open_doc(doc_id)` — full document text
- `find_in_doc(doc_id, pattern)` — regex snippets

`RetrievalEnvironment` records every tool call in the trace ledger. Eval uses that host trace for
budget scoring; model-reported traces are preserved only as debugging metadata.

## Programs

| Program | Description |
| --- | --- |
| `heuristic` | Deterministic question-only lexical search + extractive baseline |
| `single_shot_rag` | Deterministic retrieve-once top-k RAG control |
| `rlm` | DSPy RLM over corpus JSON (OpenAI-compatible API) |
| `optimized` | GEPA-tuned RLM loaded from `--artifact` |

The programs are ordered as an experiment ladder, not as interchangeable product
paths. `heuristic` validates the generated task set, gold support IDs, scorer,
and report format. `single_shot_rag` adds a conventional retrieve-once RAG
control so the RLM is compared against a stronger non-agentic baseline, not just
against lexical plumbing. `rlm` is the base policy under test: it must use
host-backed retrieval tools, and evaluation reads the host-recorded trace rather
than trusting model-reported tool use. This host-recorded trace is the audit
surface for budget and tool-use claims. `optimized` is the same policy surface
after GEPA has optimized the textual instructions/artifact.

## Metrics

Weighted composite score:

| Component | Weight |
| --- | ---: |
| answer correctness | 0.35 |
| evidence recall | 0.35 |
| citation precision | 0.15 |
| budget efficiency | 0.15 |

GEPA uses the same scorer; `retrieval_gepa_metric` returns `dspy.Prediction(score=..., feedback=...)`
with human-readable failure text (missed passages, bad citations, missing terms, wasted searches).
Predictions without an execution trace receive zero budget-efficiency credit.

## Results Protocol

Run results should be reported from eval JSON, with the same generated questions
and corpus across compared programs.

| Stage | Run | Dependency | Interpretable result |
| --- | --- | --- | --- |
| Lexical control | `eval --program heuristic` | Local deterministic code | Confirms that task generation, retrieval labels, metrics, and reports are usable |
| Retrieve-once control | `eval --program single_shot_rag` | Local deterministic code | Measures a standard top-k RAG path before agentic tool control |
| Base policy | `eval --program rlm --model ...` | OpenAI-compatible chat model | Measures the untuned DSPy RLM retrieval policy with recorded tool traces |
| Optimized policy | `optimize-gepa` then `eval --program optimized` | OpenAI-compatible chat model | Measures whether the GEPA artifact improves the same deterministic metrics |

Every public result should include:

- question generation seed and number of questions;
- corpus path or corpus commit;
- program name and model name when applicable;
- GEPA budget settings when applicable;
- mean score and component means;
- a short note for failed or skipped provider-backed runs.

Do not imply that the RLM or GEPA rows exist until those commands have been run.
The deterministic control results are still useful: they prove the task/scorer/report
loop is executable before spending provider calls.

## Architecture

```text
generate -> RetrievalExample JSONL
  -> eval (heuristic | single_shot_rag | rlm | optimized)
      -> RetrievalEnvironment + program
      -> score_retrieval
  -> optimize-gepa (optional)
      -> GEPA.compile(RLMRetrievalProgram)
      -> artifact JSON
  -> report (Markdown from run JSONs)
```

## Inspection Commands

The `demo` and `answer` commands exercise the retrieval pipeline on one question:

```text
Question -> PlannerAgent -> RetrievalAgents -> EvidenceFusion -> SynthesisAgent
```

They are useful for inspecting traces and answers during development. Use `eval`, `optimize-gepa`,
and `report` for retrieval-policy scoring and optimization.
