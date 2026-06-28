"""Observability: structured logging + Prometheus metrics.

The project declared structlog and prometheus-client but never configured them.
This module wires both: ``configure_logging`` sets up JSON structured logging
routed through the stdlib, and the Prometheus collectors + ``render_metrics``
expose an operational ``/metrics`` endpoint (request counts/latency, plus the
embedding degraded-mode counter from the LLM layer).
"""

from __future__ import annotations

import logging
import time

import structlog
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# --- Metrics (module-level singletons, registered once at import) ---------- #

HTTP_REQUESTS = Counter(
    "aikioku_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status"],
)
HTTP_REQUEST_LATENCY = Histogram(
    "aikioku_http_request_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)
EMBEDDING_DEGRADED = Gauge(
    "aikioku_embedding_deterministic_fallback_total",
    "Number of embeddings that fell back to the non-semantic deterministic hash "
    "(should be 0 in a healthy system).",
)
LLM_ERRORS = Counter(
    "aikioku_llm_errors_total",
    "Total upstream LLM/provider errors surfaced at the API boundary.",
    ["kind"],
)


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to emit JSON logs and route stdlib logging through it."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=log_level)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        cache_logger_on_first_use=True,
    )


def _normalize_path(path: str) -> str:
    """Collapse high-cardinality id segments so metric labels don't explode.

    e.g. /api/notes/<uuid> -> /api/notes/{id}. Keeps the metric cardinality bounded
    regardless of how many distinct note/entity ids are requested.
    """
    parts = path.split("/")
    out: list[str] = []
    for p in parts:
        # UUID-ish or long hex/id-looking segment -> placeholder
        if len(p) >= 16 and any(c.isdigit() for c in p) and "-" in p:
            out.append("{id}")
        else:
            out.append(p)
    return "/".join(out) or "/"


async def metrics_middleware(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Record request count + latency for every HTTP request."""
    start = time.perf_counter()
    method = request.method
    path = _normalize_path(request.url.path)
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        HTTP_REQUESTS.labels(method=method, path=path, status=500).inc()
        HTTP_REQUEST_LATENCY.labels(method=method, path=path).observe(time.perf_counter() - start)
        raise
    finally:
        # On the success path, record here (status set above). On exception the
        # except block already recorded, so guard with locals().
        if "status" in locals():
            HTTP_REQUESTS.labels(method=method, path=path, status=status).inc()
            HTTP_REQUEST_LATENCY.labels(method=method, path=path).observe(
                time.perf_counter() - start
            )


def render_metrics() -> tuple[bytes, str]:
    """Return the current Prometheus exposition payload + content type.

    Refreshes gauges sourced from other modules (e.g. the embedding fallback
    counter) so a scrape reflects live state.
    """
    try:
        from src.llm import ollama_remote

        EMBEDDING_DEGRADED.set(ollama_remote.DETERMINISTIC_FALLBACK_COUNT)
    except Exception:  # pragma: no cover - defensive
        pass
    return generate_latest(), CONTENT_TYPE_LATEST
