"""Effective embedding fingerprint.

A short, stable hash of everything that determines the embedding VECTORS. When it
changes, existing vectors are stale and the knowledge base must be reembedded.

Keyed off the dedicated EMBEDDING provider (independent of the chat provider):
- ``ollama`` → ``ollama_embedding_model`` at ``ollama_embedding_base_url``.
- ``huggingface`` → ``hf_embedding_model`` at the fixed HF inference endpoint.
- ``openai`` → ``openai_embedding_model`` at ``openai_base_url``.

``embedding_strict`` is intentionally excluded — it changes failure behavior, not
a successful vector — and is enforced by the reembed probe instead.
"""

from __future__ import annotations

import hashlib
import json

from src import runtime_config


def effective_embedding_identity() -> dict[str, str | int]:
    """Return the provider-aware identity that determines stored vectors."""
    ep = runtime_config.embedding_provider()
    if ep == "openai":
        model = runtime_config.openai_embedding_model()
        base_url = runtime_config.openai_base_url()
    elif ep == "huggingface":
        model = runtime_config.hf_embedding_model()
        base_url = "huggingface"  # fixed inference endpoint
    else:  # ollama
        model = runtime_config.ollama_embedding_model()
        base_url = runtime_config.ollama_embedding_base_url()
    return {
        "provider": ep,
        "model": model,
        "base_url": base_url,
        "dimension": runtime_config.embedding_dimension(),
    }


def effective_embedding_fingerprint() -> str:
    raw = json.dumps(effective_embedding_identity(), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
