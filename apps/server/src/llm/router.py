"""LLM cost tracking and circuit breaker with provider failover."""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when the daily LLM budget has been exceeded."""

    def __init__(self, daily_budget: float, today_cost: float) -> None:
        self.daily_budget = daily_budget
        self.today_cost = today_cost
        super().__init__(
            f"Daily LLM budget of ${daily_budget:.2f} exceeded. Current spend: ${today_cost:.2f}"
        )


@dataclass
class CostRecord:
    """Record of a single LLM call cost."""

    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CostTracker:
    """Tracks LLM usage costs and enforces daily budget caps."""

    # Rough per-token pricing (USD) — override via config if needed
    _PRICE_PER_1K: dict[str, dict[str, float]] = {
        "ollama": {"prompt": 0.0, "completion": 0.0},
        "ollama_remote": {"prompt": 0.0, "completion": 0.0},
        # OpenRouter pricing is per-model and billed upstream; left at 0.0 here.
        "openrouter": {"prompt": 0.0, "completion": 0.0},
    }

    def __init__(
        self,
        db_path: str,
        daily_budget_usd: float = 5.0,
    ) -> None:
        self._db_path = db_path
        self.daily_budget = daily_budget_usd
        self._init_db()
        self._cached_cost: float | None = None
        self._cached_cost_ts: float = 0.0

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_costs_ts ON llm_costs(timestamp)")

    def record(self, record: CostRecord) -> None:
        """Persist a cost record."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO llm_costs
                (provider, model, prompt_tokens, completion_tokens, cost_usd, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.provider,
                    record.model,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.cost_usd,
                    record.timestamp.isoformat(),
                ),
            )
        self._cached_cost = None  # invalidate

    def get_today_cost(self) -> float:
        """Return total cost for today in USD (cached, 60s TTL)."""
        if self._cached_cost is not None and (time.monotonic() - self._cached_cost_ts) < 60.0:
            return self._cached_cost
        today = datetime.now(timezone.utc).date().isoformat()
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT SUM(cost_usd) FROM llm_costs WHERE timestamp >= ?",
                (today,),
            ).fetchone()
        self._cached_cost = row[0] or 0.0
        self._cached_cost_ts = time.monotonic()
        return self._cached_cost

    def estimate_cost(self, provider: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD for a call."""
        prices = self._PRICE_PER_1K.get(provider, {"prompt": 0.0, "completion": 0.0})
        return (
            prompt_tokens / 1000.0 * prices["prompt"]
            + completion_tokens / 1000.0 * prices["completion"]
        )

    def check_budget(self, estimated_cost: float) -> bool:
        """Return True if the call is within the daily budget."""
        return (self.get_today_cost() + estimated_cost) <= self.daily_budget

    def remaining(self) -> float:
        """Budget left for today in USD (never negative)."""
        return max(0.0, self.daily_budget - self.get_today_cost())

    def fraction(self) -> float:
        """Today's spend as a fraction of the daily budget (0.0 when unlimited)."""
        # A non-positive budget means "no cap" — report 0 so callers never pause.
        if self.daily_budget <= 0:
            return 0.0
        return self.get_today_cost() / self.daily_budget

    def is_exhausted(self) -> bool:
        """True when today's spend has reached the daily budget."""
        return self.daily_budget > 0 and self.get_today_cost() >= self.daily_budget

    def is_warning(self, threshold: float) -> bool:
        """True in the near-limit band: ``threshold`` <= fraction < 1.0."""
        return self.daily_budget > 0 and threshold <= self.fraction() < 1.0

    def get_stats(self, days: int = 7) -> dict:
        """Return cost stats for the last N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT provider, SUM(cost_usd) as total_cost, COUNT(*) as call_count,
                       SUM(prompt_tokens) as total_prompt, SUM(completion_tokens) as total_completion
                FROM llm_costs WHERE timestamp >= ?
                GROUP BY provider
                """,
                (cutoff,),
            ).fetchall()
        return {
            "daily_budget": self.daily_budget,
            "today_cost": self.get_today_cost(),
            "providers": [
                {
                    "provider": r["provider"],
                    "total_cost": r["total_cost"],
                    "call_count": r["call_count"],
                    "total_prompt_tokens": r["total_prompt"],
                    "total_completion_tokens": r["total_completion"],
                }
                for r in rows
            ],
        }


@dataclass
class CircuitState:
    """Circuit breaker state for a provider."""

    failures: int = 0
    last_failure: datetime | None = None
    open_until: datetime | None = None

    def is_open(self, threshold: int = 5, cooldown_seconds: int = 60) -> bool:
        if self.failures >= threshold:
            if self.open_until is None:
                self.open_until = datetime.now(timezone.utc) + timedelta(seconds=cooldown_seconds)
            return datetime.now(timezone.utc) < self.open_until
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.open_until = None

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure = datetime.now(timezone.utc)


class LLMRouter(LLMProvider):
    """Routes LLM calls across multiple providers with circuit breaker and cost tracking."""

    def __init__(
        self,
        providers: list[LLMProvider],
        cost_tracker: CostTracker | None = None,
        failure_threshold: int = 5,
        cooldown_seconds: int = 60,
    ) -> None:
        self._providers = providers
        self._cost_tracker = cost_tracker
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._circuits: dict[int, CircuitState] = {i: CircuitState() for i in range(len(providers))}
        self._avail_ttl = float(os.environ.get("LLM_AVAILABILITY_TTL", "60"))
        self._avail_cache: dict[int, tuple[bool, float]] = {}

    async def _is_available_cached(self, idx: int, provider: LLMProvider) -> bool:
        """Cached availability check; the blocking probe runs off the event loop."""
        now = time.monotonic()
        cached = self._avail_cache.get(idx)
        if cached is not None and now - cached[1] < self._avail_ttl:
            return cached[0]
        available = await asyncio.to_thread(provider.is_available)
        self._avail_cache[idx] = (available, now)
        return available

    async def _first_available(self) -> tuple[int, LLMProvider]:
        for i, provider in enumerate(self._providers):
            if not self._circuits[i].is_open(self._failure_threshold, self._cooldown_seconds):
                if await self._is_available_cached(i, provider):
                    return i, provider
        # Fallback: return first provider even if circuit is open
        return 0, self._providers[0]

    def _record_failure(self, idx: int) -> None:
        """Record a provider failure and drop its cached availability.

        Invalidating only on failure (never on success) keeps the cache warm
        through busy successful workloads while re-probing a just-died provider.
        """
        self._circuits[idx].record_failure()
        self._avail_cache.pop(idx, None)

    async def complete(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        if self._cost_tracker is not None:
            estimated = self._cost_tracker.estimate_cost(
                type(self._providers[0]).__name__,
                len(prompt) // 4,
                len(prompt) // 4,  # rough completion estimate
            )
            if not self._cost_tracker.check_budget(estimated):
                raise BudgetExceededError(
                    self._cost_tracker.daily_budget,
                    self._cost_tracker.get_today_cost(),
                )
        idx, provider = await self._first_available()
        try:
            result = await provider.complete(prompt, system, **kwargs)
            self._circuits[idx].record_success()
            self._track(provider, prompt, result)
            return result
        except Exception as exc:
            self._record_failure(idx)
            logger.warning("provider_failed: %s - %s", type(provider).__name__, str(exc))
            # Try next provider
            for fallback_idx, fallback in enumerate(self._providers):
                if fallback_idx == idx:
                    continue
                if not self._circuits[fallback_idx].is_open(
                    self._failure_threshold, self._cooldown_seconds
                ):
                    try:
                        result = await fallback.complete(prompt, system, **kwargs)
                        self._circuits[fallback_idx].record_success()
                        self._track(fallback, prompt, result)
                        return result
                    except Exception as exc2:
                        self._record_failure(fallback_idx)
                        logger.warning(
                            "fallback_failed: %s - %s", type(fallback).__name__, str(exc2)
                        )
            raise

    async def stream(self, prompt: str, system: str = "", **kwargs: Any) -> AsyncIterator[str]:
        if self._cost_tracker is not None:
            estimated = self._cost_tracker.estimate_cost(
                type(self._providers[0]).__name__,
                len(prompt) // 4,
                len(prompt) // 4,  # rough completion estimate
            )
            if not self._cost_tracker.check_budget(estimated):
                raise BudgetExceededError(
                    self._cost_tracker.daily_budget,
                    self._cost_tracker.get_today_cost(),
                )
        idx, provider = await self._first_available()
        try:
            async for chunk in provider.stream(prompt, system, **kwargs):
                yield chunk
            self._circuits[idx].record_success()
            self._track(provider, prompt, "")
        except Exception as exc:
            self._record_failure(idx)
            logger.warning("stream_provider_failed: %s - %s", type(provider).__name__, str(exc))
            # Fallback: complete non-streaming and yield as single chunk
            for fallback_idx, fallback in enumerate(self._providers):
                if fallback_idx == idx:
                    continue
                if not self._circuits[fallback_idx].is_open(
                    self._failure_threshold, self._cooldown_seconds
                ):
                    try:
                        result = await fallback.complete(prompt, system, **kwargs)
                        self._circuits[fallback_idx].record_success()
                        self._track(fallback, prompt, result)
                        yield result
                        return
                    except Exception as exc2:
                        self._record_failure(fallback_idx)
                        logger.warning(
                            "stream_fallback_failed: %s - %s", type(fallback).__name__, str(exc2)
                        )
            raise

    async def embed(self, text: str) -> list[float]:
        idx, provider = await self._first_available()
        try:
            result = await provider.embed(text)
            self._circuits[idx].record_success()
            return result
        except Exception:
            self._record_failure(idx)
            for fallback_idx, fallback in enumerate(self._providers):
                if fallback_idx == idx:
                    continue
                if not self._circuits[fallback_idx].is_open(
                    self._failure_threshold, self._cooldown_seconds
                ):
                    try:
                        result = await fallback.embed(text)
                        self._circuits[fallback_idx].record_success()
                        return result
                    except Exception:
                        self._record_failure(fallback_idx)
            raise

    def is_available(self) -> bool:
        for i in range(len(self._providers)):
            if self._circuits[i].is_open(self._failure_threshold, self._cooldown_seconds):
                continue
            cached = self._avail_cache.get(i)
            if cached is None or cached[0]:
                return True
        return False

    def _track(self, provider: LLMProvider, prompt: str, result: str) -> None:
        if self._cost_tracker is None:
            return
        # Naive token counting (1 token ≈ 4 chars)
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(result) // 4
        cost = self._cost_tracker.estimate_cost(
            type(provider).__name__, prompt_tokens, completion_tokens
        )
        record = CostRecord(
            provider=type(provider).__name__,
            model=getattr(provider, "model", "unknown"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )
        self._cost_tracker.record(record)

    def get_circuit_states(self) -> list[dict]:
        """Return current circuit breaker states."""
        return [
            {
                "provider": type(p).__name__,
                "failures": self._circuits[i].failures,
                "open": self._circuits[i].is_open(self._failure_threshold, self._cooldown_seconds),
            }
            for i, p in enumerate(self._providers)
        ]
