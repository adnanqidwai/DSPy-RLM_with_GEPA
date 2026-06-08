# Literature Review: Trace-Grounded Retrieval Policy Harness

Review date: 2026-05-29

Scope: recent papers and engineering write-ups that motivate a compact,
provider-backed retrieval-policy harness with deterministic citation and trace
evaluation. This is not a full survey; it is a project-direction scan.

## Sources Checked

| Source | Date / venue | What matters for this project |
| --- | --- | --- |
| [MindSearch: Mimicking Human Minds Elicits Deep AI Searcher](https://arxiv.org/abs/2407.20183) | Submitted 2024-07-29, revised 2025-10-31, ICLR 2025 | Models information seeking as a dynamic graph. A WebPlanner decomposes a query into atomic sub-questions and WebSearcher agents retrieve hierarchically and in parallel. |
| [STORM: Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207) | NAACL 2024 | Uses multi-perspective question asking before writing: discover perspectives, simulate grounded conversations, then curate an outline. Useful as a retrieval-intent generator. |
| [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366) | Submitted 2025-01-09 | Inserts search when the reasoning process hits knowledge uncertainty, and uses a separate Reason-in-Documents module to refine retrieved content before injection. |
| [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136) | Submitted 2025-01-15, revised 2026-04-01 | Frames Agentic RAG around planning, reflection, tool use, multi-agent collaboration, adaptive retrieval, memory, coordination, evaluation, and governance. |
| [WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/abs/2504.21776) | Submitted 2025-04-30, revised 2025-10-13, NeurIPS 2025 | Interleaves thinking, searching, navigating, and drafting. The training recipe is too heavy, but the inference workflow is useful for a provider-backed retrieval-policy harness. |
| [MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning](https://arxiv.org/abs/2505.20096) | Submitted 2025-05-26, revised 2025-10-11 | Uses Planner, Step Definer, Extractor, and QA agents to decompose ambiguous/multi-hop QA into interpretable intermediate stages. |
| [RAGentA: Multi-Agent Retrieval-Augmented Generation for Attributed Question Answering](https://arxiv.org/abs/2506.16988) | Submitted 2025-06-20, revised 2025-09-01, SIGIR 2025 | Focuses on attributed QA. Iteratively filters documents, generates cited answers, verifies completeness, and uses hybrid sparse+dense retrieval. |
| [MemSearch-o1: Reasoning-Aligned Memory Growth in Agentic Search](https://arxiv.org/abs/2604.17265) | Submitted 2026-04-19, revised 2026-05-12 | Calls out memory dilution in iterative think-search loops and proposes fine-grained memory fragments plus retracing. Good motivation for an evidence ledger. |
| [MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](https://arxiv.org/abs/2604.18509) | Submitted 2026-04-20, ACL 2026 Findings | Uses role-specialized agents for summarization, extraction, and reasoning over retrieved documents, then synthesizes the final answer. Training-free and directly implementable. |
| [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) | Engineering blog, 2025 | Production pattern: orchestrator-worker agents, parallel subagents as intelligent filters, end-state evaluation, long-horizon memory, and filesystem/artifact handoffs to avoid information loss. |

## What Seems Current

The recent frontier is less about "add a vector database" and more about retrieval control:

- **Who asks the search questions?** Recent systems use planners, perspective agents, and uncertainty-triggered search rather than one embedding query.
- **How is evidence processed?** Evidence is not simply stuffed into a context window. Papers increasingly add extraction, reasoning, verification, and synthesis roles.
- **How is search state preserved?** Long search loops need memory/artifact stores. Otherwise, agent context becomes diluted or summaries lose source fidelity.
- **How is trust measured?** Modern systems emphasize attribution, faithfulness, retrieved-support coverage, and end-state checks, not just answer fluency.

## Project Bet

Build **RLM-GEPA Retrieval**: a DSPy RLM + GEPA harness where a model-controlled
retrieval policy explores a local corpus through host-backed tools, leaves an
execution trace, cites passage IDs, and receives deterministic failure feedback.

The project should not claim a new benchmark result. The defensible claim is:

> A trace-grounded retrieval-policy harness that compares deterministic lexical
> and retrieve-once RAG controls, a base DSPy RLM policy, and a GEPA-optimized
> policy with the same citation-support and budget metrics.

## Candidate Approaches Considered

### 1. Deep Research Clone

Implement a MindSearch/WebThinker-style web research agent with planner, web search,
navigation, and report drafting.

Pros: recognizable and current.

Cons: depends on web search APIs, page parsing, rate limits, and subjective report quality.
Harder to evaluate cleanly.

### 2. Trace-Grounded Attributed QA Harness

Implement a RAGentA/MASS-RAG-style attributed QA system over a small local
corpus, but keep the evaluation centered on host-recorded retrieval traces,
passage-level citations, and deterministic feedback.

Pros: provider-backed, Mac-friendly, measurable, and close to current
literature. Easy to show retrieval traces and citation support.

Cons: less flashy than full web research unless later extended with web tools.

### 3. Memory-Growth Agentic Search

Implement a MemSearch-o1-inspired evidence memory that grows and retraces fine-grained
fragments across iterative search rounds.

Pros: very recent and intellectually interesting.

Cons: harder to make credible without recreating parts of the paper's algorithm and eval.

## Recommendation

Start with **Approach 2** and borrow the best pieces from the others:

- MindSearch: planner-generated search graph.
- STORM: perspective-driven query expansion.
- Search-o1/WebThinker: uncertainty-triggered follow-up search.
- RAGentA: attributed QA and hybrid retrieval.
- MASS-RAG: role-specialized evidence summarizer/extractor/reasoner.
- MemSearch-o1: evidence ledger that avoids memory dilution.
- Anthropic blog: orchestrator-worker design and artifact-first subagent outputs.

This gives a current, honest, runnable project whose evaluation can be
deterministic before provider-backed RLM and GEPA runs are added.

## DSPy RLM + GEPA Fit

DSPy RLM gives the project a concrete long-context/retrieval controller: the model can use Python-style
iteration over a corpus representation and emit a trace, instead of receiving one static top-k context.
GEPA is a natural optimizer because retrieval failures are easy to explain in text: missed gold passages,
unsupported citations, incomplete answer terms, and wasted searches.

## Baseline And Results Shape

The deterministic `heuristic` program is the first result row because it checks
whether the generated questions, corpus labels, metrics, and report writer are
coherent. The deterministic `single_shot_rag` program is the second control: it
uses one top-k retrieval call and extractive synthesis, making it a better proxy
for ordinary RAG than the lexical sanity check alone. Both should be described
as controls, not as the core contribution.

The RLM and GEPA rows become meaningful only after running the provider-backed
commands. They should use the same generated task file as the baseline, and the
public report should preserve component scores so readers can see whether any
gain came from answer terms, evidence recall, citation precision, or budget
efficiency.
