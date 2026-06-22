"""Retrieval layer: dense, sparse, graph, and hybrid fusion."""

from src.retrieval.search_result import SearchResult

__all__ = [
    "SearchResult",
]

try:
    from src.retrieval.dense import DenseRetriever  # noqa: F401

    __all__.append("DenseRetriever")
except ModuleNotFoundError:
    pass

try:
    from src.retrieval.sparse import SparseRetriever  # noqa: F401

    __all__.append("SparseRetriever")
except ModuleNotFoundError:
    pass

try:
    from src.retrieval.graph_retrieval import GraphRetriever  # noqa: F401

    __all__.append("GraphRetriever")
except ModuleNotFoundError:
    pass

try:
    from src.retrieval.fusion import HybridFusion  # noqa: F401

    __all__.append("HybridFusion")
except ModuleNotFoundError:
    pass
