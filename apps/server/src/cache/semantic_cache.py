"""Semantic query cache using GPTCache with ONNX embedding + FAISS vector store.

Cache hits return instantly (<20ms); cache misses proceed normally and store the
result for next time. The cache is invalidated on any note write (create/update/delete)
since new or changed notes may alter the correct answer.

Architecture:
- GPTCache SSDataManager with ONNX embedding (fast CPU, ~10ms) + FAISS vector store
- SQLite for metadata, FAISS for vector similarity search
- All GPTCache operations run in a dedicated thread pool (SQLite/FAISS are not async-safe)
- SHA-256 cache key from query+mode+tone for deterministic identification
- Mode/tone validation on cache hit (different modes produce different answers)
- Graceful degradation: if gptcache is not installed, the cache is silently disabled

Compatibility:
- GPTCache 0.1.44+ uses a different API than earlier versions (no Cache.put/get).
  We use the SSDataManager directly (save/search/get_scalar_data/flush).
- The Onnx embedding and OnnxModelEvaluation classes use ``tokenizer.encode_plus``
  which was removed in transformers 5.x. We monkey-patch them to use the modern
  ``tokenizer(data, ...)`` API at init time.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_data_manager: Any = None
_onnx_embedding: Any = None
_similarity_eval: Any = None
_init_lock = asyncio.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semcache")
_enabled: bool = False

# ---------------------------------------------------------------------------
# Statistics (best-effort; not persisted)
# ---------------------------------------------------------------------------

_hits: int = 0
_misses: int = 0
_stats_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Environment toggles
# ---------------------------------------------------------------------------

_SIMILARITY_THRESHOLD = float(
    os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.85")
)
_TTL = int(os.getenv("SEMANTIC_CACHE_TTL", "3600"))


def _build_cache_key(query: str, mode: str, tone: str) -> str:
    """Build a deterministic SHA-256 cache key from query + mode + tone."""
    raw = f"{query}|{mode}|{tone}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _patch_onnx_for_transformers_v5() -> None:
    """Monkey-patch GPTCache Onnx classes for transformers >= 5.x compatibility.

    GPTCache 0.1.44 uses ``tokenizer.encode_plus()`` which was removed in
    transformers 5.x. We replace it with the modern ``tokenizer(data, ...)`` API.
    This is called once at init time and is idempotent.
    """
    try:
        import numpy as np
        from gptcache.embedding.onnx import Onnx as OnnxEmbed  # type: ignore[import-untyped]
        from gptcache.similarity_evaluation.onnx import (  # type: ignore[import-untyped]
            OnnxModelEvaluation as OnnxEval,
        )

        # --- Patch Onnx.to_embeddings ---
        _original_to_emb = OnnxEmbed.to_embeddings

        def _patched_to_embeddings(self, data, **_):
            encoded = self.tokenizer(
                data, padding="max_length", truncation=True, return_tensors="np"
            )
            ort_inputs = {
                "input_ids": encoded["input_ids"].astype("int64"),
                "attention_mask": encoded["attention_mask"].astype("int64"),
                "token_type_ids": np.zeros_like(encoded["input_ids"]).astype("int64"),
            }
            ort_outputs = self.ort_session.run(None, ort_inputs)
            ort_feat = ort_outputs[0]
            emb = self.post_proc(ort_feat, ort_inputs["attention_mask"])
            return emb.flatten()

        OnnxEmbed.to_embeddings = _patched_to_embeddings

        # --- Patch OnnxModelEvaluation.inference ---
        _original_inference = OnnxEval.inference

        def _patched_inference(self, reference: str, candidates: list[str]):
            n_candidates = len(candidates)
            inference_texts = [
                {"text_a": reference, "text_b": candidate} for candidate in candidates
            ]
            batch_encoding_list = [
                self.tokenizer(
                    text["text_a"], text["text_b"], padding="longest", return_tensors="np"
                )
                for text in inference_texts
            ]

            input_ids_list = [encode["input_ids"].flatten() for encode in batch_encoding_list]
            attention_mask_list = [
                encode["attention_mask"].flatten() for encode in batch_encoding_list
            ]
            token_type_ids_list = [
                encode["token_type_ids"].flatten() for encode in batch_encoding_list
            ]

            # Pad sequences to the same length
            max_len = max(len(ids) for ids in input_ids_list)

            def _pad(arr, length):
                padded = np.zeros(length, dtype=arr.dtype)
                padded[: len(arr)] = arr
                return padded

            padded_input_ids = np.stack([_pad(ids, max_len) for ids in input_ids_list])
            padded_attention_mask = np.stack(
                [_pad(mask, max_len) for mask in attention_mask_list]
            )
            padded_token_type_ids = np.stack(
                [_pad(ttids, max_len) for ttids in token_type_ids_list]
            )

            ort_inputs = {
                "input_ids": padded_input_ids.reshape(n_candidates, -1),
                "attention_mask": padded_attention_mask.reshape(n_candidates, -1),
                "token_type_ids": padded_token_type_ids.reshape(n_candidates, -1),
            }
            ort_outputs = self.ort_session.run(None, ort_inputs)
            scores = ort_outputs[0][:, 1]
            return float(scores[0])

        OnnxEval.inference = _patched_inference

        logger.debug("Patched GPTCache Onnx classes for transformers >= 5.x compatibility")

    except Exception:
        # If patching fails (e.g. gptcache not installed), that's fine —
        # the init will fail gracefully later.
        pass


async def _init_cache() -> tuple[Any, Any, Any] | None:
    """Initialize the GPTCache data manager + embedding + similarity eval.

    Returns (data_manager, onnx_embedding, similarity_eval) or None if disabled.
    Idempotent and lock-protected.
    """
    global _data_manager, _onnx_embedding, _similarity_eval, _enabled

    if _data_manager is not None:
        return (_data_manager, _onnx_embedding, _similarity_eval)

    async with _init_lock:
        if _data_manager is not None:
            return (_data_manager, _onnx_embedding, _similarity_eval)

        enabled_env = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower()
        if enabled_env not in ("true", "1", "yes"):
            _enabled = False
            logger.info(
                "Semantic cache disabled via SEMANTIC_CACHE_ENABLED=%s", enabled_env
            )
            return None

        try:
            from gptcache.embedding import Onnx  # type: ignore[import-untyped]
            from gptcache.manager import (  # type: ignore[import-untyped]
                CacheBase,
                VectorBase,
                get_data_manager,
            )
            from gptcache.similarity_evaluation import (  # type: ignore[import-untyped]
                OnnxModelEvaluation,
            )

            # Patch for transformers 5.x compatibility before creating instances
            _patch_onnx_for_transformers_v5()

            data_dir = os.path.expanduser("~/.aikioku/cache")
            os.makedirs(data_dir, exist_ok=True)

            onnx = Onnx()
            sim_eval = OnnxModelEvaluation()
            dm = get_data_manager(
                cache_base=CacheBase(
                    "sqlite", sql_url=f"sqlite:///{data_dir}/cache.db"
                ),
                vector_base=VectorBase(
                    "faiss", dimension=onnx.dimension, index_path=f"{data_dir}/faiss.index"
                ),
            )

            _data_manager = dm
            _onnx_embedding = onnx
            _similarity_eval = sim_eval
            _enabled = True
            logger.info(
                "Semantic cache initialized (threshold=%.2f, ttl=%ds, dir=%s)",
                _SIMILARITY_THRESHOLD,
                _TTL,
                data_dir,
            )
            return (dm, onnx, sim_eval)

        except ImportError:
            _enabled = False
            logger.warning(
                "gptcache not installed — semantic cache disabled. "
                "Install with: pip install gptcache onnxruntime"
            )
            return None

        except Exception:
            _enabled = False
            logger.warning(
                "Failed to initialize semantic cache — disabled.", exc_info=True
            )
            return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def cache_get(query: str, mode: str, tone: str) -> dict | None:
    """Check the semantic cache for a query matching the given mode and tone.

    Uses ONNX embedding for the query, FAISS for vector similarity search, and
    the ONNX similarity model for cross-encoding. When a cached result is found,
    the stored mode/tone are validated against the request — different modes/tone
    produce different answers, so a mismatch is treated as a cache miss.

    Returns:
        dict with keys ``response``, ``citations``, ``sub_questions``, or None.

    Never raises — failures are logged and treated as cache misses.
    """
    global _hits, _misses

    init_result = await _init_cache()
    if init_result is None:
        return None

    dm, onnx, sim_eval = init_result

    try:
        loop = asyncio.get_event_loop()

        # 1. Embed the query
        query_emb = await loop.run_in_executor(_executor, onnx.to_embeddings, query)

        # 2. Search FAISS for similar vectors
        search_results = await loop.run_in_executor(_executor, dm.search, query_emb)

        if not search_results:
            async with _stats_lock:
                _misses += 1
            return None

        # 3. For each candidate, evaluate similarity with the ONNX cross-encoder
        #    and check the similarity threshold.
        best_match: Any = None
        best_score: float = 0.0

        for distance, data_id in search_results:
            scalar_data = await loop.run_in_executor(
                _executor, dm.get_scalar_data, (distance, data_id)
            )
            if scalar_data is None or not scalar_data.answers:
                continue

            cached_question = scalar_data.question
            if not cached_question:
                continue

            # Cross-encode similarity (ONNX inference — must run in thread pool)
            score = await loop.run_in_executor(
                _executor,
                sim_eval.evaluation,
                {"question": query},
                {"question": cached_question},
            )
            if score >= _SIMILARITY_THRESHOLD and score > best_score:
                best_score = score
                best_match = scalar_data

        if best_match is None:
            async with _stats_lock:
                _misses += 1
            return None

        # 4. Deserialize and validate mode/tone
        cached_answer = best_match.answers[0].answer
        if isinstance(cached_answer, bytes):
            cached_answer = cached_answer.decode("utf-8")

        data = json.loads(cached_answer)
        if data.get("mode") != mode or data.get("tone") != tone:
            async with _stats_lock:
                _misses += 1
            return None

        async with _stats_lock:
            _hits += 1

        logger.debug(
            "Cache hit for query hash=%s (mode=%s, tone=%s, score=%.4f)",
            _build_cache_key(query, mode, tone),
            mode,
            tone,
            best_score,
        )
        return {
            "response": data["response"],
            "citations": data.get("citations", []),
            "sub_questions": data.get("sub_questions", []),
        }

    except Exception:
        logger.warning("Cache get failed — treated as miss.", exc_info=True)
        async with _stats_lock:
            _misses += 1
        return None


async def cache_put(
    query: str,
    mode: str,
    tone: str,
    response: str,
    citations: list,
    sub_questions: list,
) -> None:
    """Store a query-result pair in the semantic cache.

    Runs fire-and-forget in the chat pipeline — never blocks the response.
    Failures are logged and silently ignored.

    The stored value is a JSON dict containing the response, citations,
    sub_questions, mode, tone, and a SHA-256 key hash for debugging.
    """
    init_result = await _init_cache()
    if init_result is None:
        return

    dm, onnx, _sim_eval = init_result

    try:
        value = json.dumps(
            {
                "response": response,
                "citations": citations,
                "sub_questions": sub_questions,
                "mode": mode,
                "tone": tone,
                "key_hash": _build_cache_key(query, mode, tone),
            }
        )

        loop = asyncio.get_event_loop()
        query_emb = await loop.run_in_executor(_executor, onnx.to_embeddings, query)
        await loop.run_in_executor(_executor, dm.save, query, value, query_emb)

        logger.debug(
            "Cache stored for query hash=%s (ttl=%ds)",
            _build_cache_key(query, mode, tone),
            _TTL,
        )

    except Exception:
        logger.warning("Cache put failed — ignored.", exc_info=True)


async def cache_invalidate() -> None:
    """Flush the entire semantic cache.

    Called after any note create/update/delete because new or changed notes
    may alter the correct answer for cached queries. Runs fire-and-forget;
    never raises.
    """
    global _hits, _misses

    dm = _data_manager  # read once
    if dm is None:
        return

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, dm.flush)
        async with _stats_lock:
            _hits = 0
            _misses = 0
        logger.info("Semantic cache flushed (note write triggered invalidation)")

    except Exception:
        logger.warning("Cache flush failed — ignored.", exc_info=True)


async def cache_stats() -> dict:
    """Return cache statistics: enabled, hits, misses, hit_rate.

    Always succeeds — returns zeros when the cache is disabled.
    """
    async with _stats_lock:
        total = _hits + _misses
        hit_rate = (_hits / total) if total > 0 else 0.0
        return {
            "enabled": _enabled,
            "hits": _hits,
            "misses": _misses,
            "hit_rate": round(hit_rate, 4),
        }
