"""MemoryConsolidator — 7-stage pipeline for memory maintenance.

Stages:
  1. Cluster   — group memories by shared entities
  2. Deduplicate— remove exact/near-duplicate memories
  3. Merge     — combine complementary memories (keep highest confidence)
  4. Summarize — extract guidelines from patterns (MVP: passthrough)
  5. Conflict  — detect contradictions, flag for review
  6. Forgetting— Memory Vitality Score → tiered actions
  7. KG Sync   — update KnowledgeGraph with results
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from src.events import EventBus
from src.knowledge.entity_resolution import EntityResolver
from src.knowledge.graph import KnowledgeGraph
from src.memory.graph_sync import remove_memory_from_graph, sync_memories_to_graph
from src.models.memory import Memory, MemoryTier

if TYPE_CHECKING:
    from src.llm.base import LLMProvider

logger = structlog.get_logger()


class MemoryConsolidator:
    """Runs the 7-stage memory consolidation pipeline."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        event_bus: EventBus,
        llm_provider: "LLMProvider | None" = None,
    ) -> None:
        self._graph = graph
        self._event_bus = event_bus
        self._llm = llm_provider

    # ------------------------------------------------------------------ pipeline

    async def run(self, memories: list[Memory]) -> dict[str, Any]:
        """Run the full 7-stage pipeline.

        Returns a summary dict with counts for each stage.
        """
        input_count = len(memories)
        logger.info("consolidation.run.start", input_count=input_count)

        if not memories:
            return {
                "input_count": 0,
                "output_count": 0,
                "duplicates_removed": 0,
                "conflicts_detected": 0,
                "archived_count": 0,
                "memories": [],
            }

        # Stage 1: Cluster
        clusters = await self.stage_cluster(memories)

        # Stage 2: Deduplicate
        deduped = await self.stage_deduplicate(clusters)
        duplicates_removed = input_count - sum(len(c) for c in deduped)

        # Stage 3: Merge
        merged = await self.stage_merge(deduped)

        # Stage 4: Summarize (MVP: passthrough)
        summarized = await self.stage_summarize(merged)

        # Stage 5: Conflict Resolution
        flagged = await self.stage_conflict_resolution(summarized)
        conflicts_detected = sum(1 for m in flagged if m.confidence < 0.5)

        # Stage 6: Forgetting
        processed = await self.stage_forgetting(flagged)
        archived_count = sum(1 for m in processed if m.tier == MemoryTier.cold)

        # Stage 7: KG Sync — entities/relations for survivors, cleanup for the rest
        processed_ids = {m.id for m in processed}
        removed = [m for m in memories if m.id not in processed_ids]
        cold = [m for m in processed if m.tier == MemoryTier.cold]
        await self.stage_kg_sync(processed, removed, cold)

        output_count = len(processed)
        summary: dict[str, Any] = {
            "input_count": input_count,
            "output_count": output_count,
            "duplicates_removed": duplicates_removed,
            "conflicts_detected": conflicts_detected,
            "archived_count": archived_count,
        }
        logger.info("consolidation.run.done", **summary)
        # Surface the final processed memories so callers can persist the
        # tier/confidence updates and dedup/merge results back to storage.
        summary["memories"] = processed
        return summary

    # ------------------------------------------------------------- stage 1: Cluster

    async def stage_cluster(self, memories: list[Memory]) -> list[list[Memory]]:
        """Group memories by shared entities in subject/object.

        MVP: group by exact match on subject or object.
        """
        clusters: list[list[Memory]] = []
        assigned: set[str] = set()

        for mem in memories:
            if mem.id in assigned:
                continue
            cluster = [mem]
            assigned.add(mem.id)
            for other in memories:
                if other.id in assigned:
                    continue
                if (
                    other.subject == mem.subject
                    or other.object == mem.object
                    or other.subject == mem.object
                    or other.object == mem.subject
                ):
                    cluster.append(other)
                    assigned.add(other.id)
            clusters.append(cluster)

        logger.debug("consolidation.cluster", cluster_count=len(clusters))
        return clusters

    # ------------------------------------------------------------- stage 2: Deduplicate

    async def stage_deduplicate(self, clusters: list[list[Memory]]) -> list[list[Memory]]:
        """Remove exact duplicates based on (subject, predicate, object).

        MVP: exact match on the triple key.
        """
        result: list[list[Memory]] = []
        for cluster in clusters:
            seen: dict[tuple[str, str, str], Memory] = {}
            deduped: list[Memory] = []
            for mem in cluster:
                key = (mem.subject, mem.predicate, mem.object)
                if key not in seen:
                    seen[key] = mem
                    deduped.append(mem)
            result.append(deduped)
        return result

    # ------------------------------------------------------------- stage 3: Merge

    async def stage_merge(self, deduped: list[list[Memory]]) -> list[Memory]:
        """Merge clusters — keep highest confidence per (subject, predicate, object).

        MVP: flatten clusters and keep the highest-confidence memory for each
        unique triple.
        """
        best: dict[tuple[str, str, str], Memory] = {}
        for cluster in deduped:
            for mem in cluster:
                key = (mem.subject, mem.predicate, mem.object)
                if key not in best or mem.confidence > best[key].confidence:
                    best[key] = mem
        return list(best.values())

    # ------------------------------------------------------------- stage 4: Summarize

    async def stage_summarize(self, merged: list[Memory]) -> list[Memory]:
        """Extract guidelines from patterns using LLM-based summarization.

        Generates a summary note for each cluster of memories that share
        the same subject and groups complementary predicates.

        If no LLM provider was injected, LLM summarization is skipped and the
        merged memories are returned unchanged (rule-based passthrough).
        """
        if self._llm is None:
            return merged

        llm = self._llm

        # Group memories by subject
        subject_groups: dict[str, list[Memory]] = {}
        for mem in merged:
            subject_groups.setdefault(mem.subject, []).append(mem)

        summarized: list[Memory] = []
        for subject, mems in subject_groups.items():
            if len(mems) <= 1:
                summarized.extend(mems)
                continue

            # Build a prompt with all memories for this subject
            lines = [f"- {m.predicate} {m.object} (confidence: {m.confidence:.2f})" for m in mems]
            prompt = (
                f"Given the following facts about '{subject}', extract a concise guideline or pattern.\n"
                "Facts:\n" + "\n".join(lines) + "\n\n"
                "Guideline:"
            )
            try:
                guideline = await llm.complete(
                    prompt=prompt,
                    system="You are a pattern extraction assistant. Summarize the key insight in one sentence.",
                )
                guideline = guideline.strip()
            except Exception:
                guideline = ""

            if guideline:
                # Create a synthetic summary memory
                summary_mem = Memory(
                    subject=subject,
                    predicate="has_guideline",
                    object=guideline,
                    confidence=round(sum(m.confidence for m in mems) / len(mems), 4),
                    source="consolidation_summary",
                )
                summarized.append(summary_mem)
            summarized.extend(mems)

        return summarized

    # ------------------------------------------------------------- stage 5: Conflict

    async def stage_conflict_resolution(self, summarized: list[Memory]) -> list[Memory]:
        """Detect contradictions: same subject+predicate, different object.

        Flag conflicting memories by reducing confidence.
        """
        # Group by (subject, predicate)
        groups: dict[tuple[str, str], list[Memory]] = {}
        for mem in summarized:
            key = (mem.subject, mem.predicate)
            groups.setdefault(key, []).append(mem)

        result: list[Memory] = []
        for key, mems in groups.items():
            if len(mems) <= 1:
                result.extend(mems)
                continue

            # Check if there are different objects
            objects = {m.object for m in mems}
            if len(objects) <= 1:
                result.extend(mems)
                continue

            # Contradiction detected — flag all but the highest-confidence one
            sorted_mems = sorted(mems, key=lambda m: m.confidence, reverse=True)
            result.append(sorted_mems[0])  # keep highest as-is
            for conflict_mem in sorted_mems[1:]:
                # Reduce confidence to flag for review
                conflict_mem.confidence = min(conflict_mem.confidence, 0.4)
                result.append(conflict_mem)

        return result

    # ------------------------------------------------------------- stage 6: Forgetting

    async def stage_forgetting(self, flagged: list[Memory]) -> list[Memory]:
        """Apply Memory Vitality Score → tiered actions.

        MVP: mark low-confidence (<0.3) or old (>365 days) memories as cold.
        """
        cutoff = datetime.utcnow() - timedelta(days=365)
        for mem in flagged:
            mem_created = mem.created
            if mem_created.tzinfo is not None:
                mem_created = mem_created.replace(tzinfo=None)
            if mem.confidence < 0.3 or mem_created < cutoff:
                mem.tier = MemoryTier.cold
            elif mem.vitality_score < 0.5:
                mem.tier = MemoryTier.warm
            else:
                mem.tier = MemoryTier.hot
        return flagged

    # ------------------------------------------------------------- stage 7: KG Sync

    async def stage_kg_sync(
        self,
        all_processed: list[Memory],
        removed: list[Memory],
        cold: list[Memory],
    ) -> None:
        """Reconcile the knowledge graph with the consolidated memories.

        Survivors (non-cold) are (re)synced — entities resolved, relations upserted
        idempotently, confidence blended. Memories dropped by dedup/merge
        (``removed``) or forgotten (``cold``) have their derived relations removed
        and any sole-source entities deleted.

        Typing requires the LLM; with no provider the sync of survivors is skipped
        (mirrors ``stage_summarize``), but cleanup of removed/cold relations still
        runs.
        """
        if self._llm is not None:
            live = [m for m in all_processed if m.tier != MemoryTier.cold]
            resolver = EntityResolver(self._graph)
            await sync_memories_to_graph(live, self._graph, self._llm, resolver)

        for mem in removed + cold:
            try:
                remove_memory_from_graph(mem, self._graph)
            except Exception as e:
                logger.warning(
                    "consolidation.kg_sync.cleanup_error",
                    subject=mem.subject,
                    error=str(e),
                )
