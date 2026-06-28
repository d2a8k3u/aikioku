"""HybridFusion: weighted reciprocal rank fusion of dense, sparse, and graph retrievers."""

from __future__ import annotations

import asyncio

from collections.abc import Awaitable
from typing import TYPE_CHECKING, cast

from src.retrieval.search_result import SearchResult

if TYPE_CHECKING:
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.sparse import SparseRetriever
    from src.retrieval.graph_retrieval import GraphRetriever

# Reciprocal Rank Fusion constant
_RRF_K = 60


class HybridFusion:
    """Fuses results from dense, sparse, and graph retrievers using weighted RRF.

    Attributes:
        dense: The dense (embedding-based) retriever.
        sparse: The sparse (keyword-based) retriever.
        graph: The graph (entity-based) retriever.
        weights: A dict mapping retriever names to their fusion weights.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        graph: GraphRetriever,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Create a HybridFusion.

        Args:
            dense: The dense retriever.
            sparse: The sparse retriever.
            graph: The graph retriever.
            weights: Optional weights for each retriever. Defaults to
                {"dense": 0.4, "sparse": 0.3, "graph": 0.3}.
        """
        self.dense = dense
        self.sparse = sparse
        self.graph = graph
        self.weights = weights or {"dense": 0.4, "sparse": 0.3, "graph": 0.3}

    # Per-retriever timeouts (seconds).
    _DENSE_TIMEOUT = 5.0
    _SPARSE_TIMEOUT = 3.0
    _GRAPH_TIMEOUT = 3.0
    # Total fusion timeout — if exceeded, return whatever is available.
    _FUSION_TIMEOUT = 8.0

    async def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search using all three retrievers in parallel and fuse results.

        Each retriever has its own timeout; on timeout that retriever's results
        are empty and fusion continues with the remaining retrievers.  A total
        fusion timeout guards the whole operation.

        Args:
            query: The search query text.
            limit: Maximum number of results to return.

        Returns:
            A list of fused SearchResult objects ordered by descending score.
        """
        dense_task = self._with_timeout(self.dense.search(query, limit=20), self._DENSE_TIMEOUT)
        sparse_future = self._with_timeout(
            asyncio.to_thread(self.sparse.search, query, limit=20), self._SPARSE_TIMEOUT
        )
        graph_future = self._with_timeout(
            asyncio.to_thread(self.graph.search, query, limit=20), self._GRAPH_TIMEOUT
        )

        try:
            dense_results, sparse_results, graph_results = await asyncio.wait_for(
                asyncio.gather(dense_task, sparse_future, graph_future),
                timeout=self._FUSION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            # Total fusion timeout — return whatever partial results we have.
            # Each individual task may have already completed or timed out;
            # gather what we can from the completed tasks.
            dense_results = await self._collect(dense_task)
            sparse_results = await self._collect(sparse_future)
            graph_results = await self._collect(graph_future)

        results = {
            "dense": dense_results,
            "sparse": sparse_results,
            "graph": graph_results,
        }

        fused = self._fuse(results)
        return fused[:limit]

    @staticmethod
    async def _with_timeout(
        coro_or_future: Awaitable[list[SearchResult]], timeout: float
    ) -> list[SearchResult]:
        """Await *coro_or_future* with a timeout; return [] on timeout."""
        try:
            return await asyncio.wait_for(coro_or_future, timeout=timeout)
        except asyncio.TimeoutError:
            return []

    @staticmethod
    async def _collect(task: Awaitable[list[SearchResult]]) -> list[SearchResult]:
        """Collect a task's result if done, otherwise return []."""
        # The total-timeout branch hands us the same awaitables it created; treat
        # them as the Futures they are scheduled into to read their settled state.
        fut = cast("asyncio.Future[list[SearchResult]]", task)
        if fut.done() and not fut.cancelled():
            try:
                return fut.result()
            except Exception:
                return []
        return []

    def _fuse(self, results: dict[str, list[SearchResult]]) -> list[SearchResult]:
        """Apply weighted Reciprocal Rank Fusion.

        For each note, ACCUMULATES across retrievers:
            score = sum(weight * 1/(rank + k))
        where k=60 and weight is the retriever's fusion weight. Summing (rather
        than taking the max) is what rewards cross-retriever consensus.

        Keeps one representative SearchResult per note_id, preferring a
        non-empty snippet. Sorts by the final summed score descending.

        Args:
            results: A dict mapping retriever names to their result lists.

        Returns:
            A deduplicated, sorted list of SearchResult objects.
        """
        fused_scores: dict[str, float] = {}
        fused_results: dict[str, SearchResult] = {}

        for source, result_list in results.items():
            weight = self.weights.get(source, 0.0)
            if weight == 0.0:
                continue
            for rank, result in enumerate(result_list):
                rrf_score = weight * (1.0 / (rank + _RRF_K))
                fused_scores[result.note_id] = fused_scores.get(result.note_id, 0.0) + rrf_score

                existing = fused_results.get(result.note_id)
                if existing is None:
                    fused_results[result.note_id] = SearchResult(
                        note_id=result.note_id,
                        score=fused_scores[result.note_id],
                        source="fusion",
                        snippet=result.snippet,
                        metadata=result.metadata,
                        source_type=result.source_type,
                    )
                else:
                    existing.score = fused_scores[result.note_id]
                    # Upgrade to a non-empty snippet (and its metadata) if the
                    # representative is still empty and this source has content.
                    # source_type is NOT overwritten here — it keeps whatever was
                    # set when the result was first added (preserves "entity" for
                    # synthetic entity results).
                    if not existing.snippet and result.snippet:
                        existing.snippet = result.snippet
                        if result.metadata:
                            existing.metadata = result.metadata

        # Sort by fused score descending
        sorted_results = sorted(fused_results.values(), key=lambda r: r.score, reverse=True)
        return sorted_results
