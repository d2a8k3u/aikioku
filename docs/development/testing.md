# Testing

## Backend (pytest)

89 test files spanning every subsystem. Integration tests are excluded by
default (`addopts` carries `-m 'not integration'`).

```bash
cd apps/server
source .venv/bin/activate

# Default suite (integration excluded)
pytest

# Include integration tests
pytest -m integration

# Skip slow tests (LLM calls, large datasets)
pytest -m "not slow"

# Specific layer
pytest tests/test_knowledge/
pytest tests/test_retrieval/
pytest tests/test_api/

# Coverage
pytest --cov=src --cov-report=term-missing
```

### Test Structure

| Directory | Files | Focus |
|-----------|-------|-------|
| `tests/test_api/` | 28 | REST endpoint integration |
| `tests/test_knowledge/` | 10 | Graph, embeddings, entity resolution |
| `tests/test_integration/` | 7 | End-to-end pipelines |
| `tests/test_models/` | 6 | Note, Entity, Relation, Memory, Card, User |
| `tests/test_retrieval/` | 5 | Dense, sparse, fusion, graph retrieval |
| `tests/test_reasoning/` | 5 | RAG, connections, multi-hop |
| `tests/test_augmentation/` | 5 | Spaced repetition, summarization, serendipity |
| `tests/test_llm/` | 5 | Provider abstraction and parsing utilities |
| `tests/test_memory/` | 4 | Extraction, consolidation, tiering |
| `tests/test_ingestion/` | 2 | Parsers and ingestion API |
| `tests/test_storage/` | 2 | Note store, note index |
| `tests/test_plugins/` | 1 | Plugin manager |

### Markers

```python
@pytest.mark.unit        # Unit tests
@pytest.mark.integration # Integration tests (excluded by default)
@pytest.mark.slow        # Slow tests (LLM calls, large datasets)
```

### Config

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers -m 'not integration'"
pythonpath = ["."]
```

## Frontend (vitest)

```bash
cd apps/dashboard
npm test
```

### Config

```typescript
// vitest.config.ts
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/__tests__/setup.ts"],
  },
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
});
```

Both `pytest` and `vitest run` exit non-zero on failure, so they drop into CI
unchanged.
