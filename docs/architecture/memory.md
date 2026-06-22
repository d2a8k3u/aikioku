
# Layer 4: Memory

Extracts, stores, and consolidates knowledge over time — mimicking human memory
systems.

## Extraction

LLM-driven episodic memory extraction from notes:

1. Note content → LLM prompt
2. LLM identifies discrete facts, insights, events
3. Each memory linked to source note and entities
4. Memories stored with confidence score

## Tiering

Three-tier system based on access patterns and recency:

| Tier | Description | Retention | Access Pattern |
|------|-------------|-----------|----------------|
| Hot  | Recent, frequently accessed | Full detail | Immediate |
| Warm | Moderate relevance, less recent | Summarized | On-demand |
| Cold | Archival, rarely accessed | Metadata only | Search-only |

### Promotion/Demotion

- **Hot → Warm:** Not accessed for 7 days
- **Warm → Cold:** Not accessed for 30 days
- **Cold → Warm:** Accessed via search
- **Warm → Hot:** Accessed 3+ times in 24 hours

## Consolidation

Periodic background process:

1. Identify related memories across notes
2. Merge overlapping facts
3. Resolve contradictions (flag for review)
4. Decay stale memories (reduce confidence)
5. Strengthen frequently corroborated memories

## Memory Model

```python
class Memory:
    id: str
    content: str           # The remembered fact/insight
    tier: MemoryTier       # hot, warm, cold
    confidence: float      # 0.0–1.0
    source_note_id: str
    entity_ids: list[str]
    created_at: datetime
    last_accessed: datetime
    access_count: int
```
