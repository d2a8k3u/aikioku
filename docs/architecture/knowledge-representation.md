
# Layer 2: Knowledge Representation

Stores knowledge in three complementary indices for different retrieval strategies.

## Triple Store

| Store    | Technology | Purpose                              | Query Style     |
|----------|------------|--------------------------------------|-----------------|
| Graph    | Kuzu       | Entities, relations, property graph  | Cypher          |
| Dense    | ChromaDB   | Semantic similarity search           | Vector distance |
| Sparse   | Tantivy    | Full-text keyword search             | BM25            |

## Graph (Kuzu)

Embedded property graph database. No separate server — runs in-process.

- **Entities:** typed nodes with properties (name, type, description, source notes)
- **Relations:** typed edges between entities (RELATES_TO, CONTAINS, CITES, etc.)
- **Schema induction:** LLM-driven discovery of entity types and relation patterns
- **Entity resolution:** Deduplication and merging across notes

### Entity Model

```python
class Entity:
    id: str
    name: str
    type: EntityType      # PERSON, CONCEPT, TOOL, PROJECT, etc.
    properties: dict      # Flexible key-value metadata
    source_note_ids: list[str]
```

### Relation Model

```python
class Relation:
    id: str
    source_entity_id: str
    target_entity_id: str
    type: RelationType    # RELATES_TO, CONTAINS, CITES, DEPENDS_ON, etc.
    weight: float
    evidence: str         # Source text snippet
```

## Dense (ChromaDB)

Embedded vector database for semantic similarity.

- **Configurable embedding provider:** Ollama, HuggingFace, OpenAI
- **Dimension:** 1024 (default, provider-dependent)
- **Index:** HNSW with cosine distance
- **Metadata:** note ID, title, tags stored alongside vectors

## Sparse (Tantivy)

Rust-based full-text search engine with Python bindings.

- **Tokenizer:** Unicode-aware with stemming
- **Fields:** title, content, tags
- **Scoring:** BM25 with field boosting (title > content)
- **Invalidation:** Automatic on note update/delete

## Extraction Pipeline

On note creation/update:

1. LLM extracts entities and relations from note content
2. Entity resolver merges duplicates across existing entities
3. Graph updated with new nodes and edges
4. Dense embedding computed and stored
5. Sparse index updated
