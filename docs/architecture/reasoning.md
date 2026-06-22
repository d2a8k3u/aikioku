
# Layer 5: Reasoning & Generation

AI-powered reasoning over the knowledge base.

## RAG (Retrieval-Augmented Generation)

Grounded question answering with source citations:

1. User query → hybrid retrieval
2. Retrieved notes + query → LLM prompt
3. LLM generates answer with inline citations
4. Citations link back to source notes

### Modes

- **Simple:** Single retrieval pass, direct answer
- **Multi-hop:** Iterative retrieval with reasoning steps for complex questions

## Connection Discovery

Finds indirect relationships between entities:

1. Start from entity A
2. Graph traversal (BFS up to max_distance)
3. Embedding similarity check on connected entities
4. Return discovered connections with evidence paths

## Question Generation

Auto-generates review questions from notes:

1. Note content → LLM prompt
2. LLM generates factual, conceptual, and application questions
3. Questions stored as spaced-repetition cards

## Anomaly Detection

Identifies inconsistencies in the knowledge graph:

- Contradictory relations (A → B and A → ¬B)
- Orphan entities (no connections)
- Duplicate entities (high similarity, not merged)
- Stale entities (no updates in 90+ days)
