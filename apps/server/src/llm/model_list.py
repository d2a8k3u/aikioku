"""Dynamic provider model listing.

Fetches available models from Ollama (local/remote) and OpenRouter so the
settings UI can offer searchable dropdowns instead of free-text inputs.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.llm.urls import OLLAMA_TAGS, OPENROUTER_MODELS, join

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama model listing
# ---------------------------------------------------------------------------

async def list_ollama_models(
    base_url: str = "http://localhost:11434",
    api_key: str = "",
) -> list[dict[str, Any]]:
    """Fetch available models from an Ollama instance via GET /api/tags.

    Returns a list of dicts with keys: id, name, family, parameter_size.
    On error returns an empty list (caller should check and surface the error).
    """
    url = join(base_url, OLLAMA_TAGS, dialect="ollama")
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch Ollama models from %s: %s", url, exc)
        return []

    models_raw: list[dict[str, Any]] = data.get("models", [])
    result: list[dict[str, Any]] = []
    for m in models_raw:
        name = m.get("name", m.get("model", ""))
        details = m.get("details", {})
        family = details.get("family", "")
        parameter_size = details.get("parameter_size", "")
        result.append({
            "id": name,
            "name": name,
            "family": family,
            "parameter_size": parameter_size,
        })
    return result


def classify_ollama_models(
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tag each Ollama model as 'chat' or 'embedding' based on family.

    bert / nomic-bert families → embedding; everything else → chat.
    """
    EMBEDDING_FAMILIES = {"bert", "nomic-bert"}
    tagged: list[dict[str, Any]] = []
    for m in models:
        family = (m.get("family") or "").lower()
        mtype = "embedding" if family in EMBEDDING_FAMILIES else "chat"
        tagged.append({**m, "type": mtype})
    return tagged


# ---------------------------------------------------------------------------
# OpenRouter model listing
# ---------------------------------------------------------------------------

async def list_openrouter_models(
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: str = "",
) -> list[dict[str, Any]]:
    """Fetch available models from OpenRouter via GET /models.

    Returns a list of dicts with keys: id, name, context_length,
    pricing_prompt, pricing_completion.
    On error returns an empty list.
    """
    url = join(base_url, OPENROUTER_MODELS, dialect="openrouter")
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch OpenRouter models from %s: %s", url, exc)
        return []

    models_raw: list[dict[str, Any]] = data.get("data", [])
    result: list[dict[str, Any]] = []
    for m in models_raw:
        result.append({
            "id": m.get("id", ""),
            "name": m.get("name", m.get("id", "")),
            "context_length": m.get("context_length"),
            "pricing_prompt": float(m.get("pricing", {}).get("prompt", 0) or 0),
            "pricing_completion": float(m.get("pricing", {}).get("completion", 0) or 0),
        })
    return result


# ---------------------------------------------------------------------------
# HuggingFace embedding model search (public Hub API; no auth required)
# ---------------------------------------------------------------------------

async def list_hf_embedding_models(
    query: str = "", limit: int = 20
) -> list[dict[str, Any]]:
    """Search the public HuggingFace Hub for sentence-similarity (embedding) models.

    Returns a list of dicts with keys: id, name, provider, dimensions.
    Empty ``query`` lists the most-downloaded models. Empty list on error.
    """
    params = {
        "pipeline_tag": "sentence-similarity",
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
    }
    if query:
        params["search"] = query
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://huggingface.co/api/models", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch HF embedding models: %s", exc)
        return []

    result: list[dict[str, Any]] = []
    for m in data or []:
        mid = m.get("id") or m.get("modelId") or ""
        if mid:
            result.append({"id": mid, "name": mid, "provider": "huggingface", "dimensions": None})
    return result
