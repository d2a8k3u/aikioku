
# Architecture

Aikioku is built as a 7-layer system. Each layer is independently testable
and communicates through well-defined interfaces.

1. [Ingestion](./ingestion.md) — Content capture and normalization
2. [Knowledge Representation](./knowledge-representation.md) — Graph + vectors + full-text
3. [Retrieval](./retrieval.md) — Multi-strategy search with fusion
4. [Memory](./memory.md) — Extraction, consolidation, tiering
5. [Reasoning](./reasoning.md) — RAG, connections, question generation
6. [Cognitive Augmentation](./cognitive-augmentation.md) — Spaced repetition, serendipity
7. [Interface](./interface.md) — Web UI and API surface
