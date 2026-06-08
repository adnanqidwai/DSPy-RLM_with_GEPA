# Evidence Ledger

RLM-GEPA Retrieval stores retrieved passages in an evidence ledger. Each ledger row includes the
source document, passage ID, retrieving agent role, rank, fused score, support-token overlap,
provenance metadata, and citation label.

The ledger is designed to solve a memory problem in long retrieval loops. Instead of copying
large passages through every agent message, workers write durable artifacts and the coordinator
passes compact references to synthesis and verification stages.

This makes retrieval auditable. A developer can inspect which role found each passage, which
passages were fused, and whether the generated answer cites the selected evidence.

