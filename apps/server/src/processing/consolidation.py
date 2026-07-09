"""Core consolidation logic: similarity detection, clustering, LLM merge, CRUD orchestration."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI

from src import runtime_config
from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.note_cascade import purge_note_artifacts
from src.knowledge.pipeline import extract_entities_from_note, store_note_embeddings
from src.llm.base import LLMProvider
from src.llm.json_parse import LLMOutputParseError, parse_llm_json
from src.llm.long_text import condense_for_prompt
from src.llm.router import BudgetExceededError
from src.models.note import Note
from src.storage.note_store import NoteStore

logger = logging.getLogger(__name__)

_SINGLE_PASS_CHAR_BUDGET = 4000

_MERGE_SYSTEM_PROMPT = (
    "You are a knowledge consolidation assistant. Your task is to merge multiple similar notes "
    "into a single comprehensive note that preserves every piece of information from all source "
    'notes. Never omit a fact, name, number, date, or claim. Remove only exact duplicate '
    'sentences. Return your answer as JSON: {"title": "...", "content": "..."}.'
)


def _build_merge_prompt(notes: list[Note], combined_notes: str) -> str:
    return (
        f"You are merging {len(notes)} similar notes into a single comprehensive note.\n"
        "Preserve ALL information from every note — do not omit any fact, name, number, "
        "date, or claim.\n"
        "Remove only exact duplicates.\n"
        'Return JSON with keys "title" and "content".\n\n'
        f"{combined_notes}"
    )


def fetch_candidates(store: NoteStore, limit: int) -> list[Note]:
    return store.list(source_type="hidden", limit=limit)


def build_entity_map(graph: KnowledgeGraph) -> dict[str, set[str]]:
    all_entities = graph.find_entities(limit=100_000)
    note_to_entities: dict[str, set[str]] = {}
    for entity in all_entities:
        for note_id in entity.source_note_ids:
            note_to_entities.setdefault(note_id, set()).add(entity.id)
    return note_to_entities


def detect_similar_pairs(
    notes: list[Note],
    embeddings: dict[str, list[float]],
    entity_map: dict[str, set[str]],
    threshold: float,
) -> list[tuple[Note, Note, float]]:
    if len(notes) < 2:
        return []

    ids = [n.id for n in notes]
    embeddable = [(n, nid) for n, nid in zip(notes, ids) if nid in embeddings]
    if len(embeddable) < 2:
        return []

    embeddable_notes = [e[0] for e in embeddable]
    embeddable_ids = [e[1] for e in embeddable]
    vectors = np.array([embeddings[nid] for nid in embeddable_ids])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / safe_norms
    sim_matrix = normalized @ normalized.T

    pairs: list[tuple[Note, Note, float]] = []
    for i in range(len(embeddable_ids)):
        for j in range(i + 1, len(embeddable_ids)):
            sim = float(sim_matrix[i, j])
            if sim < threshold:
                continue
            entities_i = entity_map.get(embeddable_ids[i], set())
            entities_j = entity_map.get(embeddable_ids[j], set())
            if not (entities_i & entities_j):
                continue
            pairs.append((embeddable_notes[i], embeddable_notes[j], sim))

    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def _split_component(
    comp_notes: list[Note],
    pairs: list[tuple[Note, Note, float]],
    cap: int,
) -> list[list[Note]]:
    comp_ids = {n.id for n in comp_notes}
    comp_pairs = [
        (a, b, s) for a, b, s in pairs if a.id in comp_ids and b.id in comp_ids
    ]
    if not comp_pairs:
        return [comp_notes] if len(comp_notes) >= 2 else []

    assigned: dict[str, int] = {}
    sub_clusters: list[list[Note]] = []

    for a, b, _ in comp_pairs:
        a_assigned = a.id in assigned
        b_assigned = b.id in assigned
        if a_assigned and b_assigned:
            continue
        if not a_assigned and not b_assigned:
            cluster_idx = len(sub_clusters)
            sub_clusters.append([a, b])
            assigned[a.id] = cluster_idx
            assigned[b.id] = cluster_idx
        elif a_assigned:
            cluster_idx = assigned[a.id]
            if len(sub_clusters[cluster_idx]) < cap and b.id not in assigned:
                sub_clusters[cluster_idx].append(b)
                assigned[b.id] = cluster_idx
        elif b_assigned:
            cluster_idx = assigned[b.id]
            if len(sub_clusters[cluster_idx]) < cap and a.id not in assigned:
                sub_clusters[cluster_idx].append(a)
                assigned[a.id] = cluster_idx

    for note in comp_notes:
        if note.id not in assigned:
            placed = False
            for cluster in sub_clusters:
                if len(cluster) < cap:
                    cluster.append(note)
                    assigned[note.id] = sub_clusters.index(cluster)
                    placed = True
                    break
            if not placed:
                sub_clusters.append([note])
                assigned[note.id] = len(sub_clusters) - 1

    return [sc for sc in sub_clusters if len(sc) >= 2]


def cluster_pairs(
    pairs: list[tuple[Note, Note, float]],
    notes: list[Note],
    min_size: int,
    cap: int,
) -> list[list[Note]]:
    adj: dict[str, set[str]] = {n.id: set() for n in notes}
    for a, b, _ in pairs:
        adj[a.id].add(b.id)
        adj[b.id].add(a.id)

    visited: set[str] = set()
    components: list[list[str]] = []
    for note in notes:
        if note.id in visited:
            continue
        component: list[str] = []
        queue = [note.id]
        while queue:
            nid = queue.pop()
            if nid in visited:
                continue
            visited.add(nid)
            component.append(nid)
            queue.extend(adj[nid] - visited)
        components.append(component)

    note_by_id = {n.id: n for n in notes}
    result: list[list[Note]] = []
    for comp in components:
        if len(comp) < min_size:
            continue
        comp_notes = [note_by_id[nid] for nid in comp if nid in note_by_id]
        if len(comp_notes) <= cap:
            result.append(comp_notes)
        else:
            result.extend(_split_component(comp_notes, pairs, cap))
    return result


async def merge_cluster(app: FastAPI, cluster: list[Note]) -> Note | None:
    llm_provider: LLMProvider = app.state.llm_provider

    blocks: list[str] = []
    for i, note in enumerate(cluster, 1):
        content = note.content
        if len(content) > _SINGLE_PASS_CHAR_BUDGET:
            content = await condense_for_prompt(llm_provider, content)
        blocks.append(f'Note {i} (title: "{note.title}"):\n{content}')

    combined = "\n\n".join(blocks)
    prompt = _build_merge_prompt(cluster, combined)

    response = await llm_provider.complete(prompt=prompt, system=_MERGE_SYSTEM_PROMPT)

    try:
        parsed = parse_llm_json(response, expect="dict")
    except LLMOutputParseError:
        logger.warning("consolidation.merge_parse_failed: %s", [n.id for n in cluster])
        return None

    if not isinstance(parsed, dict):
        logger.warning("consolidation.merge_not_dict: %s", [n.id for n in cluster])
        return None

    title = str(parsed.get("title", "")).strip()
    content = str(parsed.get("content", "")).strip()
    if not title or not content:
        logger.warning("consolidation.merge_empty_output: %s", [n.id for n in cluster])
        return None

    all_tags: set[Any] = set()
    all_links: set[str] = set()
    for note in cluster:
        all_tags.update(note.frontmatter.get("tags", []))
        all_links.update(note.links)

    merged_note = Note(
        title=title,
        content=content,
        frontmatter={"tags": sorted(all_tags)} if all_tags else {},
        links=sorted(all_links),
        path=f"/notes/hidden/{uuid.uuid4()}.md",
        source_type="hidden",
    )
    return merged_note


def create_merged_note(store: NoteStore, merged_note: Note) -> None:
    store.create(merged_note)


def delete_originals(
    store: NoteStore,
    embedding_store: EmbeddingStore,
    graph: KnowledgeGraph,
    cluster: list[Note],
) -> None:
    for note in cluster:
        store.delete(note.id)
        purge_note_artifacts(note.id, embedding_store=embedding_store, graph=graph)


async def reprocess_merged_note(app: FastAPI, merged_note: Note) -> None:
    await extract_entities_from_note(
        merged_note, app.state.llm_provider, app.state.knowledge_graph
    )
    await store_note_embeddings(
        merged_note, app.state.embedding_provider, app.state.embedding_store
    )
    fusion = getattr(app.state, "hybrid_fusion", None)
    sparse = getattr(fusion, "sparse", None)
    if sparse is not None and hasattr(sparse, "mark_dirty"):
        sparse.mark_dirty()


async def run_consolidation(app: FastAPI) -> dict[str, int]:
    store: NoteStore = app.state.note_store
    embedding_store: EmbeddingStore = app.state.embedding_store
    graph: KnowledgeGraph = app.state.knowledge_graph

    candidates = fetch_candidates(store, runtime_config.consolidation_max_notes_per_run())
    if len(candidates) < runtime_config.consolidation_min_cluster_size():
        logger.info("consolidation.no_work: candidates=%d", len(candidates))
        return {
            "candidates": len(candidates),
            "clusters": 0,
            "merged": 0,
            "deleted": 0,
            "skipped": 0,
        }

    note_ids = [n.id for n in candidates]
    embeddings = embedding_store.get_embeddings(note_ids)
    embeddable = [n for n in candidates if n.id in embeddings]
    if len(embeddable) < runtime_config.consolidation_min_cluster_size():
        logger.info(
            "consolidation.no_embeddable: total=%d with_embeddings=%d",
            len(candidates),
            len(embeddable),
        )
        return {
            "candidates": len(candidates),
            "clusters": 0,
            "merged": 0,
            "deleted": 0,
            "skipped": 0,
        }

    entity_map = build_entity_map(graph)
    threshold = runtime_config.consolidation_similarity_threshold()
    pairs = detect_similar_pairs(embeddable, embeddings, entity_map, threshold)

    min_size = runtime_config.consolidation_min_cluster_size()
    cap = runtime_config.consolidation_cluster_size_cap()
    clusters = cluster_pairs(pairs, embeddable, min_size, cap)

    merged_count = 0
    deleted_count = 0
    skipped_count = 0
    for cluster in clusters:
        try:
            merged_note = await merge_cluster(app, cluster)
            if merged_note is None:
                skipped_count += 1
                continue
            create_merged_note(store, merged_note)
            delete_originals(store, embedding_store, graph, cluster)
            await reprocess_merged_note(app, merged_note)
            merged_count += 1
            deleted_count += len(cluster)
        except BudgetExceededError:
            logger.warning(
                "consolidation.budget_exceeded_mid_run: completed=%d remaining_clusters=%d",
                merged_count,
                len(clusters) - merged_count - skipped_count,
            )
            break
        except Exception:
            logger.exception("consolidation.cluster_failed: %s", [n.id for n in cluster])
            skipped_count += 1
            continue

    return {
        "candidates": len(candidates),
        "clusters": len(clusters),
        "merged": merged_count,
        "deleted": deleted_count,
        "skipped": skipped_count,
    }