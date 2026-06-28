"""Memory tier management: hot/warm/cold."""

from __future__ import annotations

from src.models.memory import Memory, MemoryTier


class MemoryTierManager:
    """Manages hot/warm/cold tiers for Memory objects."""

    def __init__(self) -> None:
        self._memories: list[Memory] = []

    def register(self, memory: Memory) -> None:
        """Track a memory for tier statistics and consolidation scheduling."""
        self._memories.append(memory)

    def promote(self, memory: Memory) -> None:
        """Move memory up one tier (cold→warm, warm→hot)."""
        if memory.tier == MemoryTier.cold:
            memory.tier = MemoryTier.warm
        elif memory.tier == MemoryTier.warm:
            memory.tier = MemoryTier.hot

    def demote(self, memory: Memory) -> None:
        """Move memory down one tier (hot→warm, warm→cold)."""
        if memory.tier == MemoryTier.hot:
            memory.tier = MemoryTier.warm
        elif memory.tier == MemoryTier.warm:
            memory.tier = MemoryTier.cold

    def get_tier_stats(self) -> dict[str, int]:
        """Return counts per tier among registered memories."""
        stats: dict[str, int] = {"hot": 0, "warm": 0, "cold": 0}
        for mem in self._memories:
            stats[mem.tier.value] += 1
        return stats

    def schedule_for_consolidation(self) -> list[Memory]:
        """Return memories in the cold tier as candidates for consolidation."""
        return [mem for mem in self._memories if mem.tier == MemoryTier.cold]
