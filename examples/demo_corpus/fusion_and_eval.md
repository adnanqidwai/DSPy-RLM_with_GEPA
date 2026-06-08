# Fusion and Evaluation

RLM-GEPA Retrieval fuses ranked lists with reciprocal-rank fusion. A passage receives credit when
it appears near the top of an agent's result list, and it receives a small diversity bonus
when multiple retrieval roles independently find it.

The first evaluation target is evidence support recall. For each demo question, gold support
document IDs define which documents should be retrieved. This keeps the MVP deterministic and
avoids requiring an LLM judge.

Additional metrics include cited passage count, unsupported citation count, required claim-term
coverage, and whether the answer uses citations from the evidence ledger.

