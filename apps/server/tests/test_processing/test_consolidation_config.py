"""Tests for consolidation config getters in runtime_config."""

from __future__ import annotations

from src import runtime_config


class TestConsolidationConfigDefaults:
    """Each getter returns its code-level default when no app_setting is set."""

    def test_auto_note_consolidation_defaults_false(self):
        assert runtime_config.auto_note_consolidation() is False

    def test_consolidation_similarity_threshold_default(self):
        assert runtime_config.consolidation_similarity_threshold() == 0.85

    def test_consolidation_min_cluster_size_default(self):
        assert runtime_config.consolidation_min_cluster_size() == 2

    def test_consolidation_max_notes_per_run_default(self):
        assert runtime_config.consolidation_max_notes_per_run() == 50

    def test_consolidation_cluster_size_cap_default(self):
        assert runtime_config.consolidation_cluster_size_cap() == 5

    def test_consolidation_min_budget_remaining_usd_default(self):
        assert runtime_config.consolidation_min_budget_remaining_usd() == 0.50

    def test_consolidation_interval_hours_default(self):
        assert runtime_config.consolidation_interval_hours() == 6.0


class TestConsolidationConfigTypes:
    """Getters return values of the declared type."""

    def test_auto_note_consolidation_is_bool(self):
        assert isinstance(runtime_config.auto_note_consolidation(), bool)

    def test_similarity_threshold_is_float(self):
        assert isinstance(runtime_config.consolidation_similarity_threshold(), float)

    def test_min_cluster_size_is_int(self):
        assert isinstance(runtime_config.consolidation_min_cluster_size(), int)

    def test_max_notes_per_run_is_int(self):
        assert isinstance(runtime_config.consolidation_max_notes_per_run(), int)

    def test_cluster_size_cap_is_int(self):
        assert isinstance(runtime_config.consolidation_cluster_size_cap(), int)

    def test_min_budget_remaining_is_float(self):
        assert isinstance(
            runtime_config.consolidation_min_budget_remaining_usd(), float
        )

    def test_interval_hours_is_float(self):
        assert isinstance(runtime_config.consolidation_interval_hours(), float)


class TestConsolidationConfigOverrides:
    """Setting an app_setting overrides the code-level default."""

    def test_auto_note_consolidation_can_be_enabled(self):
        runtime_config.set_app_setting("auto_note_consolidation", "true")
        assert runtime_config.auto_note_consolidation() is True

    def test_similarity_threshold_override(self):
        runtime_config.set_app_setting("consolidation_similarity_threshold", "0.92")
        assert runtime_config.consolidation_similarity_threshold() == 0.92

    def test_min_cluster_size_override(self):
        runtime_config.set_app_setting("consolidation_min_cluster_size", "3")
        assert runtime_config.consolidation_min_cluster_size() == 3

    def test_max_notes_per_run_override(self):
        runtime_config.set_app_setting("consolidation_max_notes_per_run", "25")
        assert runtime_config.consolidation_max_notes_per_run() == 25

    def test_cluster_size_cap_override(self):
        runtime_config.set_app_setting("consolidation_cluster_size_cap", "8")
        assert runtime_config.consolidation_cluster_size_cap() == 8

    def test_min_budget_remaining_override(self):
        runtime_config.set_app_setting(
            "consolidation_min_budget_remaining_usd", "1.25"
        )
        assert runtime_config.consolidation_min_budget_remaining_usd() == 1.25

    def test_interval_hours_override(self):
        runtime_config.set_app_setting("consolidation_interval_hours", "12.0")
        assert runtime_config.consolidation_interval_hours() == 12.0

    def test_invalid_int_falls_back_to_default(self):
        runtime_config.set_app_setting("consolidation_min_cluster_size", "not-an-int")
        assert runtime_config.consolidation_min_cluster_size() == 2

    def test_invalid_float_falls_back_to_default(self):
        runtime_config.set_app_setting(
            "consolidation_similarity_threshold", "not-a-float"
        )
        assert runtime_config.consolidation_similarity_threshold() == 0.85