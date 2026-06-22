"""LLM provider factory.

Single source of truth for building the configured ``LLMRouter`` from
``runtime_config``. Both app startup (``main.build_runtime``) and ad-hoc
consumers (e.g. the memory API's fallback path) use this so providers are
constructed identically — never a bare localhost default that is unreachable
in-container.
"""
from __future__ import annotations

from src import runtime_config
from src.llm.base import LLMProvider
from src.llm.ollama import OllamaProvider
from src.llm.ollama_remote import OllamaRemoteProvider
from src.llm.openai_embeddings import OpenAIEmbeddingProvider
from src.llm.openrouter import OpenRouterProvider
from src.llm.router import CostTracker, LLMRouter


def _build_primary_provider(strict_embeddings: bool | None = None) -> LLMProvider:
    """Build the configured PRIMARY provider (no router, no fallback).

    ``strict_embeddings`` overrides the configured strictness when set — used by
    reembed to force a single, consistent embedding model that raises on failure
    instead of writing a deterministic/fallback (different-space) vector.
    """
    provider = runtime_config.llm_provider()
    model = runtime_config.llm_model()
    base_url = runtime_config.ollama_base_url()
    strict = runtime_config.embedding_strict() if strict_embeddings is None else strict_embeddings
    if provider == "ollama_remote":
        return OllamaRemoteProvider(
            base_url=base_url,
            model=model,
            api_key=runtime_config.ollama_api_key(),
            embedding_base_url=runtime_config.ollama_embedding_base_url(),
            embedding_model=runtime_config.ollama_embedding_model(),
            embedding_provider=runtime_config.embedding_provider(),
            strict_embeddings=strict,
            embedding_fallback_dim=runtime_config.embedding_dimension(),
            hf_api_key=runtime_config.hf_api_key(),
        )
    if provider == "openrouter":
        return OpenRouterProvider(
            api_key=runtime_config.openrouter_api_key(),
            model=model,
            base_url=runtime_config.openrouter_base_url(),
            embedding_base_url=runtime_config.ollama_embedding_base_url(),
            embedding_model=runtime_config.ollama_embedding_model(),
            embedding_provider=runtime_config.embedding_provider(),
            strict_embeddings=strict,
            embedding_fallback_dim=runtime_config.embedding_dimension(),
            hf_api_key=runtime_config.hf_api_key(),
        )
    return OllamaProvider(base_url=base_url, model=model)


def build_llm_provider(cost_tracker: CostTracker | None = None) -> LLMProvider:
    """Build the configured primary provider plus a local-Ollama fallback,
    wrapped in an ``LLMRouter`` with circuit breaker + cost tracking."""
    provider = runtime_config.llm_provider()
    base_url = runtime_config.ollama_base_url()
    model = runtime_config.llm_model()
    providers: list[LLMProvider] = [_build_primary_provider()]
    # Always add local Ollama as a fallback if not already the primary.
    if provider != "ollama":
        providers.append(OllamaProvider(base_url=base_url, model=model))
    return LLMRouter(providers, cost_tracker=cost_tracker)


def build_embedding_provider(strict: bool | None = None) -> LLMProvider:
    """Build the dedicated embedding provider from the embedding settings.

    Independent of the chat provider: query embedding, document embedding, and
    bulk reembed all go through this so the vector space is consistent regardless
    of which chat model is configured. ``strict=None`` uses the configured
    ``embedding_strict``; reembed passes ``strict=True`` so a failed call raises
    (and is retried/aborted) instead of writing a deterministic hash vector.
    """
    ep = runtime_config.embedding_provider()
    strict_eff = runtime_config.embedding_strict() if strict is None else strict
    dim = runtime_config.embedding_dimension()
    if ep == "openai":
        return OpenAIEmbeddingProvider(
            base_url=runtime_config.openai_base_url(),
            api_key=runtime_config.openai_api_key(),
            model=runtime_config.openai_embedding_model(),
            embedding_fallback_dim=dim,
            strict_embeddings=strict_eff,
        )
    # ollama (local) | ollama_remote | huggingface — reuse OllamaRemoteProvider as
    # a pure embedder (its chat base_url is unused on the embedding path).
    if ep == "huggingface":
        model = runtime_config.hf_embedding_model()
        embed_routing = "huggingface"
        embed_key = ""
    else:  # ollama (local) or ollama_remote — both use the native /api/embed path
        model = runtime_config.ollama_embedding_model()
        embed_routing = "ollama"
        embed_key = runtime_config.ollama_api_key() if ep == "ollama_remote" else ""
    return OllamaRemoteProvider(
        base_url="",
        embedding_base_url=runtime_config.ollama_embedding_base_url(),
        embedding_model=model,
        embedding_provider=embed_routing,
        embedding_api_key=embed_key,
        hf_api_key=runtime_config.hf_api_key(),
        strict_embeddings=strict_eff,
        embedding_fallback_dim=dim,
    )
