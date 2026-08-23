"""Migrate embeddings from ChromaDB collections to the numpy-backed store.

The import-vectors mode reads every stored vector directly from the old chroma
collections and writes them into the new store unchanged — exact by
construction, no re-embedding, no quality drift. The reembed mode is
deliberately not implemented here: the in-app reembed pipeline
(src/knowledge/reembed.py) already covers recomputation when the embedding
config changes, and import-vectors is the lossless path for a store swap.

Usage (from apps/server/ inside the container):
    python -m scripts.migrate_embeddings_chroma_to_numpy \
        --old-notes-path /data/sqlite/chroma \
        --old-convs-path /data/sqlite/chroma_conversations \
        --new-notes-path /data/sqlite/emb_notes \
        --new-convs-path /data/sqlite/emb_convs

Requires the migration extra: pip install -e '.[migration]'
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import statistics
import sys
import time
from typing import Any

import numpy as np

logger = logging.getLogger("migrate_embeddings")

_BATCH = 500
_PARITY_QUERIES = 20
_PARITY_K = 20
_LATENCY_SAMPLES = 20
_LATENCY_RATIO_LIMIT = 2.0


def _read_chroma_docs(
    coll: Any,
) -> tuple[list[str], list[list[float]], list[dict[str, Any]]]:
    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []
    offset = 0
    while True:
        raw = coll.get(limit=_BATCH, offset=offset, include=["embeddings", "metadatas"])
        batch_ids: list[str] = raw["ids"]
        if not batch_ids:
            break
        ids.extend(batch_ids)
        for emb in raw["embeddings"]:
            embeddings.append([float(v) for v in emb])
        metadatas.extend(dict(m) for m in raw["metadatas"])
        if len(batch_ids) < _BATCH:
            break
        offset += len(batch_ids)
    return ids, embeddings, metadatas


def _old_top_k(coll: Any, query_embeddings: list[list[float]], k: int) -> list[list[str]]:
    total = coll.count()
    results: list[list[str]] = []
    for q in query_embeddings:
        raw = coll.query(
            query_embeddings=[q],
            n_results=total,
            include=["metadatas", "distances"],
        )
        dists = raw["distances"][0]
        metas = raw["metadatas"][0]
        best: dict[str, float] = {}
        for distance, meta in zip(dists, metas):
            note_id = str(meta.get("note_id", ""))
            if not note_id:
                continue
            score = 1.0 - float(distance)
            if score <= 0:
                continue
            if note_id not in best or score > best[note_id]:
                best[note_id] = score
        results.append(
            [nid for nid, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        )
    return results


def _embed_queries(queries: list[str]) -> list[list[float]]:
    from src.llm.factory import build_embedding_provider

    async def _run() -> list[list[float]]:
        provider = build_embedding_provider(strict=True)
        try:
            return [await provider.embed(q) for q in queries]
        finally:
            if hasattr(provider, "aclose"):
                await provider.aclose()

    return asyncio.run(_run())


def _make_fixture(path: str) -> None:
    import chromadb

    rng = np.random.default_rng(42)
    for dir_path, coll_name in (
        (os.path.join(path, "notes"), "notes__fixturefp"),
        (os.path.join(path, "convs"), "conversations__fixturefp"),
    ):
        client = chromadb.PersistentClient(path=dir_path)
        coll = client.create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine", "embedding_dimension": 64},
        )
        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        for i in range(30):
            vec = rng.standard_normal(64)
            vec = vec / float(np.linalg.norm(vec))
            note_id = f"note-{i:03d}"
            for chunk in range(2):
                ids.append(f"{note_id}#{chunk}")
                embeddings.append([float(v) for v in vec])
                metadatas.append(
                    {"note_id": note_id, "chunk_index": chunk, "text": f"note {i} chunk {chunk} fixture text"}
                )
        coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)  # type: ignore[arg-type]
    logger.info("migration.fixture.done path=%s", path)


def _backup(path: str, backup_dir: str) -> None:
    if not os.path.isdir(path):
        return
    stamp = time.strftime("%Y%m%d")
    dest = os.path.join(backup_dir, f"{os.path.basename(path.rstrip('/'))}-{stamp}")
    if os.path.exists(dest):
        logger.info("migration.backup.skip exists=%s", dest)
        return
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copytree(path, dest)
    logger.info("migration.backup.done src=%s dst=%s", path, dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-notes-path")
    parser.add_argument("--old-convs-path")
    parser.add_argument("--new-notes-path")
    parser.add_argument("--new-convs-path")
    parser.add_argument("--notes-collection", default=None)
    parser.add_argument("--convs-collection", default=None)
    parser.add_argument("--mode", choices=["import-vectors", "reembed"], default="import-vectors")
    parser.add_argument("--queries", default="")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--make-fixture", metavar="PATH")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.make_fixture:
        _make_fixture(args.make_fixture)
        return 0

    for label, value in (
        ("old-notes-path", args.old_notes_path),
        ("old-convs-path", args.old_convs_path),
        ("new-notes-path", args.new_notes_path),
        ("new-convs-path", args.new_convs_path),
    ):
        if not value:
            logger.error("migration.arg.missing %s", label)
            return 2

    if args.mode == "reembed":
        logger.error(
            "reembed mode is not implemented here; use --mode import-vectors (exact, "
            "lossless) or the in-app reembed pipeline (src/knowledge/reembed.py)"
        )
        return 2

    logger.info("migration.gate_determinism.skip import-vectors is exact by construction")

    import chromadb

    from src.knowledge.embeddings import EmbeddingStore

    old_client = chromadb.PersistentClient(path=args.old_notes_path)
    notes_coll = old_client.get_collection(
        name=args.notes_collection or old_client.list_collections()[0].name
    )
    conv_client = chromadb.PersistentClient(path=args.old_convs_path)
    convs_coll = conv_client.get_collection(
        name=args.convs_collection or conv_client.list_collections()[0].name
    )
    logger.info("migration.collections notes=%s convs=%s", notes_coll.name, convs_coll.name)

    notes_ids, notes_embs, notes_metas = _read_chroma_docs(notes_coll)
    convs_ids, convs_embs, convs_metas = _read_chroma_docs(convs_coll)
    logger.info("migration.read notes_docs=%d convs_docs=%d", len(notes_ids), len(convs_ids))

    if not notes_ids and not convs_ids:
        logger.error("migration.read.empty both collections are empty")
        return 2

    dim = len(notes_embs[0]) if notes_embs else len(convs_embs[0])

    if args.dry_run:
        logger.info(
            "migration.dry_run would write notes_docs=%d convs_docs=%d dim=%d",
            len(notes_ids),
            len(convs_ids),
            dim,
        )
        logger.info("migration.dry_run gates skipped")
        return 0

    backup_dir = os.path.abspath(
        args.backup_dir or os.path.join(os.path.dirname(args.old_notes_path), "..", "migration-backups")
    )
    _backup(args.old_notes_path, backup_dir)
    _backup(args.old_convs_path, backup_dir)

    new_notes = EmbeddingStore(args.new_notes_path, collection_name="notes_migrated", dimension=dim)
    for doc_id, emb, meta in zip(notes_ids, notes_embs, notes_metas):
        new_notes.add_document(
            doc_id=doc_id,
            text=str(meta.get("text", "")),
            embedding=emb,
            metadata={
                "note_id": str(meta.get("note_id", "")),
                "chunk_index": int(meta.get("chunk_index", 0)),
            },
        )
    logger.info("migration.import.notes.docs=%d", new_notes.count())

    new_convs = EmbeddingStore(args.new_convs_path, collection_name="conversations_migrated", dimension=dim)
    for doc_id, emb, meta in zip(convs_ids, convs_embs, convs_metas):
        new_convs.add_document(
            doc_id=doc_id,
            text=str(meta.get("text", "")),
            embedding=emb,
            metadata={
                "note_id": str(meta.get("note_id", "")),
                "chunk_index": int(meta.get("chunk_index", 0)),
            },
        )
    logger.info("migration.import.convs.docs=%d", new_convs.count())

    if args.synthetic:
        query_embs = notes_embs[:_PARITY_QUERIES]
    else:
        queries = [" ".join(str(m.get("text", "")).split()[:8]) for m in notes_metas[:_PARITY_QUERIES]]
        queries = [q for q in queries if q]
        extra = [q.strip() for q in args.queries.split(",") if q.strip()]
        queries = (queries + extra)[:_PARITY_QUERIES]
        query_embs = _embed_queries(queries) if queries else []

    if not query_embs:
        logger.warning("migration.gate_parity.skip no queries available")
    else:
        old_ranks = _old_top_k(notes_coll, query_embs, _PARITY_K)
        mismatches = 0
        for i, (old_rank, q) in enumerate(zip(old_ranks, query_embs)):
            new_rank = [r["note_id"] for r in new_notes.search(q, _PARITY_K)]
            if old_rank != new_rank:
                mismatches += 1
                logger.error(
                    "migration.gate_parity.fail query=%d old=%s new=%s", i, old_rank[:5], new_rank[:5]
                )
        logger.info("migration.gate_parity.pass=%d/%d", len(old_ranks) - mismatches, len(old_ranks))
        if mismatches:
            return 2

    q = query_embs[0] if query_embs else notes_embs[0]
    old_times: list[float] = []
    for _ in range(_LATENCY_SAMPLES):
        t0 = time.perf_counter()
        _old_top_k(notes_coll, [q], 20)
        old_times.append((time.perf_counter() - t0) * 1000)
    new_times: list[float] = []
    for _ in range(_LATENCY_SAMPLES):
        t0 = time.perf_counter()
        new_notes.search(q, 20)
        new_times.append((time.perf_counter() - t0) * 1000)
    old_med = statistics.median(old_times)
    new_med = statistics.median(new_times)
    logger.info("migration.gate_latency old_ms=%.3f new_ms=%.3f", old_med, new_med)
    if new_med > _LATENCY_RATIO_LIMIT * old_med:
        logger.error("migration.gate_latency.fail")
        return 2

    logger.info("migration.done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
