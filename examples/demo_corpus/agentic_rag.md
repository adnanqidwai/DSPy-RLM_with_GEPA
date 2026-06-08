# Agentic RAG

Static RAG usually runs one retrieval query, inserts the top passages into a prompt, and
asks a model to answer. Agentic RAG instead gives the system control over retrieval:
it can plan, call tools, reformulate questions, reflect on missing evidence, and run
multiple retrieval steps before producing an answer.

Multi-agent retrieval is useful when one query hides several intents. A planner can split
the question into subquestions, while worker agents search for direct facts, background
context, counterevidence, implementation details, and evaluation criteria.

The main risk is that extra agents can amplify noise. A trustworthy system needs evidence
provenance, source-aware synthesis, and verification that final claims are grounded in
retrieved passages rather than in the model's memory.

