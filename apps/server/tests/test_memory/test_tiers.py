"""Tests for memory tier management."""

from __future__ import annotations


from src.models.memory import Memory, MemoryTier


class TestMemoryTierManager:
    def test_import(self):
        pass

    def test_promote_warm_to_hot(self):
        from src.memory.tiers import MemoryTierManager

        mgr = MemoryTierManager()
        mem = Memory(subject="S", predicate="P", object="O", source="src", tier=MemoryTier.warm)
        mgr.promote(mem)
        assert mem.tier == MemoryTier.hot

    def test_promote_cold_to_warm(self):
        from src.memory.tiers import MemoryTierManager

        mgr = MemoryTierManager()
        mem = Memory(subject="S", predicate="P", object="O", source="src", tier=MemoryTier.cold)
        mgr.promote(mem)
        assert mem.tier == MemoryTier.warm

    def test_promote_hot_unchanged(self):
        from src.memory.tiers import MemoryTierManager

        mgr = MemoryTierManager()
        mem = Memory(subject="S", predicate="P", object="O", source="src", tier=MemoryTier.hot)
        mgr.promote(mem)
        assert mem.tier == MemoryTier.hot

    def test_demote_hot_to_warm(self):
        from src.memory.tiers import MemoryTierManager

        mgr = MemoryTierManager()
        mem = Memory(subject="S", predicate="P", object="O", source="src", tier=MemoryTier.hot)
        mgr.demote(mem)
        assert mem.tier == MemoryTier.warm

    def test_demote_warm_to_cold(self):
        from src.memory.tiers import MemoryTierManager

        mgr = MemoryTierManager()
        mem = Memory(subject="S", predicate="P", object="O", source="src", tier=MemoryTier.warm)
        mgr.demote(mem)
        assert mem.tier == MemoryTier.cold

    def test_demote_cold_unchanged(self):
        from src.memory.tiers import MemoryTierManager

        mgr = MemoryTierManager()
        mem = Memory(subject="S", predicate="P", object="O", source="src", tier=MemoryTier.cold)
        mgr.demote(mem)
        assert mem.tier == MemoryTier.cold

    def test_get_tier_stats_empty(self):
        from src.memory.tiers import MemoryTierManager

        mgr = MemoryTierManager()
        stats = mgr.get_tier_stats()
        assert stats == {"hot": 0, "warm": 0, "cold": 0}

    def test_get_tier_stats_with_memories(self):
        from src.memory.tiers import MemoryTierManager

        mgr = MemoryTierManager()
        mgr.register(
            Memory(subject="A", predicate="is", object="A", source="src", tier=MemoryTier.hot)
        )
        mgr.register(
            Memory(subject="B", predicate="is", object="B", source="src", tier=MemoryTier.warm)
        )
        mgr.register(
            Memory(subject="C", predicate="is", object="C", source="src", tier=MemoryTier.cold)
        )
        mgr.register(
            Memory(subject="D", predicate="is", object="D", source="src", tier=MemoryTier.hot)
        )
        stats = mgr.get_tier_stats()
        assert stats == {"hot": 2, "warm": 1, "cold": 1}

    def test_schedule_for_consolidation_returns_cold(self):
        from src.memory.tiers import MemoryTierManager

        mgr = MemoryTierManager()
        cold = Memory(subject="X", predicate="is", object="X", source="src", tier=MemoryTier.cold)
        warm = Memory(subject="Y", predicate="is", object="Y", source="src", tier=MemoryTier.warm)
        mgr.register(cold)
        mgr.register(warm)
        scheduled = mgr.schedule_for_consolidation()
        assert scheduled == [cold]

    def test_schedule_for_consolidation_empty_when_no_cold(self):
        from src.memory.tiers import MemoryTierManager

        mgr = MemoryTierManager()
        mgr.register(
            Memory(subject="Y", predicate="is", object="Y", source="src", tier=MemoryTier.warm)
        )
        assert mgr.schedule_for_consolidation() == []
