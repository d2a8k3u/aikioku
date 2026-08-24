"""Knowledge package: graph, entity resolution, and embedding storage."""

from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.entity_resolution import EntityResolver
from src.knowledge.graph import KnowledgeGraph

__all__ = ["EmbeddingStore", "EntityResolver", "KnowledgeGraph"]
