"""Cognitive State Awareness system.

Tracks behavioural signals (typing speed, deletion rate, switch rate)
and classifies the user's current cognitive state to determine
appropriate AI intervention levels.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from src.events import EventBus


class CognitiveState(Enum):
    """Classified cognitive states derived from behavioural signals."""

    FLOW = "flow"
    THINKING = "thinking"
    EXPLORING = "exploring"
    FRUSTRATED = "frustrated"
    IDLE = "idle"


class InterventionType(Enum):
    """Type of AI intervention appropriate for a given cognitive state."""

    BACKGROUND_ONLY = "background_only"  # Process but don't interrupt
    SUGGEST = "suggest"  # Gentle suggestions OK
    FULL = "full"  # All interventions OK
    PROACTIVE = "proactive"  # Offer help
    DEFER = "defer"  # Queue for later


# Mapping from cognitive state to recommended intervention type
_STATE_TO_INTERVENTION: dict[CognitiveState, InterventionType] = {
    CognitiveState.FLOW: InterventionType.BACKGROUND_ONLY,
    CognitiveState.THINKING: InterventionType.SUGGEST,
    CognitiveState.EXPLORING: InterventionType.FULL,
    CognitiveState.FRUSTRATED: InterventionType.PROACTIVE,
    CognitiveState.IDLE: InterventionType.DEFER,
}

# Classification thresholds
_FLOW_TYPING_MIN = 3.0  # chars/sec
_FLOW_TYPING_MAX = 15.0  # chars/sec
_FLOW_DELETION_MAX = 0.15  # 15%
_FLOW_SWITCH_MAX = 0.1  # minimal switching
_FLOW_DURATION_MIN = 120  # 2 minutes in seconds

_FRUSTRATED_DELETION_MIN = 0.30  # high deletion
_FRUSTRATED_TYPING_MAX = 3.0  # rapid short inputs (low speed)

_IDLE_THRESHOLD = 300  # 5 minutes in seconds

_EXPLORING_SWITCH_MIN = 0.3  # high switching
_EXPLORING_ACTIVITY_VARIETY = 3  # number of different signal types


class CognitiveStateTracker:
    """Tracks behavioural signals and classifies cognitive state.

    Signals are stored in a SQLite database (separate from the EventBus DB)
    and analysed over a sliding window to determine the user's current
    cognitive state and the appropriate AI intervention.
    """

    def __init__(self, event_bus: EventBus, db_path: Optional[str] = None) -> None:
        """Initialize the tracker.

        Args:
            event_bus: The system event bus (used for publishing state-change events).
            db_path: Optional path for the signals SQLite database.
                     If not provided, a temporary database is created.
        """
        self._event_bus = event_bus
        if db_path is None:
            # Use a temp file so each tracker instance gets its own DB by default
            import tempfile

            self._db_path = str(Path(tempfile.mkdtemp()) / "cognitive_signals.db")
        else:
            self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the signals table if it does not exist."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    signal_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)")

    def record_signal(
        self,
        signal_type: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Store a behavioural signal.

        Args:
            signal_type: Category of signal (e.g. 'typing_speed', 'deletion_rate', 'switch_rate').
            value: Numeric value of the signal.
            timestamp: When the signal occurred. Defaults to now (UTC).
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        signal_id = str(uuid4())
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO signals (id, signal_type, value, timestamp) VALUES (?, ?, ?, ?)",
                (signal_id, signal_type, value, timestamp.isoformat()),
            )

    def get_state(self, window_seconds: int = 30) -> CognitiveState:
        """Classify the current cognitive state from recent signals.

        Analyses all signals within the sliding window [now - window_seconds, now]
        and returns the best-matching CognitiveState.

        Args:
            window_seconds: Size of the analysis window in seconds.

        Returns:
            The classified CognitiveState.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - __import__("datetime").timedelta(seconds=window_seconds)
        cutoff_str = cutoff.isoformat()

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT signal_type, value, timestamp FROM signals WHERE timestamp >= ? ORDER BY timestamp ASC",
                (cutoff_str,),
            ).fetchall()

        if not rows:
            # Check if there are ANY signals at all; if the most recent one
            # is older than the idle threshold, return IDLE.
            with sqlite3.connect(self._db_path) as conn:
                latest = conn.execute(
                    "SELECT timestamp FROM signals ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
            if latest is None:
                return CognitiveState.IDLE
            latest_ts = datetime.fromisoformat(latest[0])
            if (now - latest_ts).total_seconds() > _IDLE_THRESHOLD:
                return CognitiveState.IDLE
            # Signals exist but are outside the window — still idle for this window
            return CognitiveState.IDLE

        # Aggregate signals by type
        typing_speeds = []
        deletion_rates = []
        switch_rates = []
        all_types: set[str] = set()

        for signal_type, value, _ts in rows:
            all_types.add(signal_type)
            if signal_type == "typing_speed":
                typing_speeds.append(value)
            elif signal_type == "deletion_rate":
                deletion_rates.append(value)
            elif signal_type == "switch_rate":
                switch_rates.append(value)

        avg_typing = sum(typing_speeds) / len(typing_speeds) if typing_speeds else 0.0
        avg_deletion = sum(deletion_rates) / len(deletion_rates) if deletion_rates else 0.0
        avg_switch = sum(switch_rates) / len(switch_rates) if switch_rates else 0.0

        # Compute duration of the signal window
        first_ts = datetime.fromisoformat(rows[0][2])
        last_ts = datetime.fromisoformat(rows[-1][2])
        duration_seconds = (last_ts - first_ts).total_seconds()

        # --- Classification rules (order matters: most specific first) ---

        # FRUSTRATED: high deletion, rapid short inputs
        if avg_deletion >= _FRUSTRATED_DELETION_MIN and avg_typing <= _FRUSTRATED_TYPING_MAX:
            return CognitiveState.FRUSTRATED

        # FLOW: sustained typing (3-15 chars/sec), low deletion, minimal switching, 2+ min
        if (
            typing_speeds
            and _FLOW_TYPING_MIN <= avg_typing <= _FLOW_TYPING_MAX
            and avg_deletion < _FLOW_DELETION_MAX
            and avg_switch <= _FLOW_SWITCH_MAX
            and duration_seconds >= _FLOW_DURATION_MIN
        ):
            return CognitiveState.FLOW

        # EXPLORING: high switching, varied activity
        if avg_switch >= _EXPLORING_SWITCH_MIN and len(all_types) >= _EXPLORING_ACTIVITY_VARIETY:
            return CognitiveState.EXPLORING

        # FRUSTRATED (broader): high deletion even without low typing
        if avg_deletion >= _FRUSTRATED_DELETION_MIN:
            return CognitiveState.FRUSTRATED

        # THINKING: moderate typing, moderate pauses (default for active but not matching above)
        if typing_speeds and avg_typing > 0:
            return CognitiveState.THINKING

        # Fallback
        return CognitiveState.IDLE

    def get_intervention_recommendation(self, state: CognitiveState) -> InterventionType:
        """Return the appropriate AI intervention type for a cognitive state.

        Args:
            state: The current cognitive state.

        Returns:
            The recommended InterventionType.
        """
        return _STATE_TO_INTERVENTION.get(state, InterventionType.DEFER)

    def get_signal_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent signals.

        Args:
            limit: Maximum number of signals to return.

        Returns:
            A list of dicts with keys: id, signal_type, value, timestamp.
        """
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, signal_type, value, timestamp FROM signals ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        # Return in chronological order (oldest first)
        results = [
            {
                "id": row[0],
                "signal_type": row[1],
                "value": row[2],
                "timestamp": row[3],
            }
            for row in reversed(rows)
        ]
        return results
