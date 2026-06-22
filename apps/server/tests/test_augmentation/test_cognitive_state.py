"""Tests for CognitiveStateTracker."""
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from src.events import EventBus


@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database and EventBus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_events.db")
        bus = EventBus(db_path=db_path)
        yield bus


def test_get_state_returns_flow_for_sustained_typing(tmp_db):
    """FLOW: sustained typing (3-15 chars/sec), low deletion (<15%), minimal switching, 2+ min duration."""
    from src.augmentation.cognitive_state import CognitiveStateTracker, CognitiveState

    tracker = CognitiveStateTracker(event_bus=tmp_db)
    now = datetime.now(timezone.utc)

    # Simulate 3 minutes of sustained typing at ~8 chars/sec, low deletion
    for i in range(180):  # 3 minutes, one signal per second
        ts = now + timedelta(seconds=i)
        tracker.record_signal("typing_speed", 8.0, timestamp=ts)
        tracker.record_signal("deletion_rate", 0.05, timestamp=ts)
        tracker.record_signal("switch_rate", 0.01, timestamp=ts)

    state = tracker.get_state(window_seconds=180)
    assert state == CognitiveState.FLOW


def test_get_state_returns_idle_for_no_recent_signals(tmp_db):
    """IDLE: no signals for >5 minutes."""
    from src.augmentation.cognitive_state import CognitiveStateTracker, CognitiveState

    tracker = CognitiveStateTracker(event_bus=tmp_db)
    # No signals recorded at all
    state = tracker.get_state(window_seconds=30)
    assert state == CognitiveState.IDLE


def test_get_state_returns_frustrated_for_high_deletion(tmp_db):
    """FRUSTRATED: high deletion, rapid short inputs."""
    from src.augmentation.cognitive_state import CognitiveStateTracker, CognitiveState

    tracker = CognitiveStateTracker(event_bus=tmp_db)
    now = datetime.now(timezone.utc)

    # Simulate high deletion rate and rapid short inputs
    for i in range(30):
        ts = now - timedelta(seconds=30 - i)
        tracker.record_signal("typing_speed", 2.0, timestamp=ts)
        tracker.record_signal("deletion_rate", 0.6, timestamp=ts)
        tracker.record_signal("switch_rate", 0.5, timestamp=ts)

    state = tracker.get_state(window_seconds=30)
    assert state == CognitiveState.FRUSTRATED


def test_intervention_for_flow_is_background_only(tmp_db):
    """FLOW → BACKGROUND_ONLY."""
    from src.augmentation.cognitive_state import (
        CognitiveStateTracker,
        CognitiveState,
        InterventionType,
    )

    tracker = CognitiveStateTracker(event_bus=tmp_db)
    intervention = tracker.get_intervention_recommendation(CognitiveState.FLOW)
    assert intervention == InterventionType.BACKGROUND_ONLY


def test_intervention_for_frustrated_is_proactive(tmp_db):
    """FRUSTRATED → PROACTIVE."""
    from src.augmentation.cognitive_state import (
        CognitiveStateTracker,
        CognitiveState,
        InterventionType,
    )

    tracker = CognitiveStateTracker(event_bus=tmp_db)
    intervention = tracker.get_intervention_recommendation(CognitiveState.FRUSTRATED)
    assert intervention == InterventionType.PROACTIVE


def test_signal_history_returns_recorded_signals(tmp_db):
    """get_signal_history returns the signals we recorded."""
    from src.augmentation.cognitive_state import CognitiveStateTracker

    tracker = CognitiveStateTracker(event_bus=tmp_db)
    now = datetime.now(timezone.utc)

    tracker.record_signal("typing_speed", 10.0, timestamp=now)
    tracker.record_signal("deletion_rate", 0.1, timestamp=now + timedelta(seconds=1))
    tracker.record_signal("switch_rate", 0.05, timestamp=now + timedelta(seconds=2))

    history = tracker.get_signal_history(limit=10)
    assert len(history) == 3
    assert history[0]["signal_type"] == "typing_speed"
    assert history[0]["value"] == 10.0
    assert history[1]["signal_type"] == "deletion_rate"
    assert history[2]["signal_type"] == "switch_rate"
