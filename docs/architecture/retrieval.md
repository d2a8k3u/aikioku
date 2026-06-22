
# Layer 3: Retrieval

Multi-strategy retrieval with reciprocal rank fusion for best recall.

## Strategies

### Dense Retrieval
- ChromaDB vector similarity search
- Configurable embedding provider (Ollama, HuggingFace, OpenAI)
- Returns top-k by cosine distance
- Best for: semantic/conceptual queries

### Sparse Retrieval
- Tantivy BM25 full-text search
- Field-weighted (title boosted over content)
- Returns top-k by relevance score
- Best for: keyword/exact-match queries

### Graph Retrieval
- Kuzu Cypher queries
- BFS subgraph expansion around entities
- Path finding between entities
- Best for: relational/structural queries

## Hybrid Fusion

Reciprocal Rank Fusion (RRF) combines all three strategies:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where `k = 60` (standard constant) and `rank_i` is the document's rank in each
result list.

### Fusion Pipeline

1. Execute dense, sparse, and graph queries in parallel
2. Normalize scores per strategy
3. Apply RRF formula
4. Sort by fused score
5. Return top-k with source annotations

## Multi-Hop Retrieval

Iterative retrieval for complex questions:

1. Initial query → retrieve candidate notes
2. Extract entities from candidates
3. Graph traversal from extracted entities
4. Retrieve connected notes
5. Repeat until convergence or max hops

## Search Result Model

```python
class SearchResult:
    note_id: str
    title: str
    snippet: str         # Relevant excerpt
    score: float         # Fused relevance score
    source: str          # "dense", "sparse", "graph", "fusion"
    entities: list[str]  # Related entity names
```
