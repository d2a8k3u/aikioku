"""Anomaly detection for knowledge quality and system behavior."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of an anomaly detection check."""

    type: str
    severity: str  # low, medium, high, critical
    description: str
    entity_id: str | None = None
    note_id: str | None = None
    metric_value: float | None = None
    threshold: float | None = None


class AnomalyDetector:
    """Detects anomalies in knowledge graph, memory store, and system metrics."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS anomalies (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    entity_id TEXT,
                    note_id TEXT,
                    metric_value REAL,
                    threshold REAL,
                    detected_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_type ON anomalies(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_resolved ON anomalies(resolved)")

    def record(self, result: AnomalyResult) -> None:
        """Persist an anomaly to the database."""
        import uuid

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO anomalies
                (id, type, severity, description, entity_id, note_id, metric_value, threshold, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    result.type,
                    result.severity,
                    result.description,
                    result.entity_id,
                    result.note_id,
                    result.metric_value,
                    result.threshold,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_recent(self, hours: int = 24, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent unresolved anomalies."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM anomalies WHERE detected_at >= ? AND resolved = 0 ORDER BY detected_at DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def resolve(self, anomaly_id: str) -> None:
        """Mark an anomaly as resolved."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("UPDATE anomalies SET resolved = 1 WHERE id = ?", (anomaly_id,))

    def detect_low_confidence_entities(self, graph: KnowledgeGraph) -> list[AnomalyResult]:
        """Find entities with very low confidence scores."""
        results: list[AnomalyResult] = []
        try:
            entities = graph.find_entities(limit=10000)
        except Exception:
            return results
        for entity in entities:
            if entity.confidence < 0.3:
                results.append(
                    AnomalyResult(
                        type="low_confidence_entity",
                        severity="medium",
                        description=f"Entity '{entity.name}' has low confidence ({entity.confidence:.2f})",
                        entity_id=entity.id,
                        metric_value=entity.confidence,
                        threshold=0.3,
                    )
                )
        return results

    def detect_isolated_entities(self, graph: KnowledgeGraph) -> list[AnomalyResult]:
        """Find entities with no relations."""
        results: list[AnomalyResult] = []
        try:
            entities = graph.find_entities(limit=10000)
        except Exception:
            return results
        for entity in entities:
            try:
                relations = graph.get_relations(entity.id)
            except Exception:
                continue
            if len(relations) == 0:
                results.append(
                    AnomalyResult(
                        type="isolated_entity",
                        severity="low",
                        description=f"Entity '{entity.name}' has no relations",
                        entity_id=entity.id,
                        metric_value=0.0,
                        threshold=1.0,
                    )
                )
        return results

    def detect_memory_decay(self, db_path: str) -> list[AnomalyResult]:
        """Find memories with very low vitality scores."""
        results: list[AnomalyResult] = []
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT id, subject, vitality_score FROM memories WHERE vitality_score < 0.2"
                ).fetchall()
        except Exception:
            return results
        for row in rows:
            results.append(
                AnomalyResult(
                    type="memory_decay",
                    severity="low",
                    description=f"Memory '{row[1]}' has very low vitality ({row[2]:.2f})",
                    metric_value=row[2],
                    threshold=0.2,
                )
            )
        return results

    def detect_card_backlog(self, db_path: str) -> list[AnomalyResult]:
        """Find large numbers of due cards indicating review backlog."""
        results: list[AnomalyResult] = []
        try:
            with sqlite3.connect(db_path) as conn:
                now = datetime.utcnow().isoformat()
                due_count = conn.execute(
                    "SELECT COUNT(*) FROM cards WHERE next_review <= ?", (now,)
                ).fetchone()[0]
                total = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        except Exception:
            return results
        if total > 0 and due_count / total > 0.5:
            results.append(
                AnomalyResult(
                    type="card_backlog",
                    severity="high",
                    description=f"{due_count} of {total} cards are due for review (>50%)",
                    metric_value=due_count,
                    threshold=total * 0.5,
                )
            )
        elif due_count > 100:
            results.append(
                AnomalyResult(
                    type="card_backlog",
                    severity="medium",
                    description=f"{due_count} cards are due for review",
                    metric_value=due_count,
                    threshold=100.0,
                )
            )
        return results

    def run_all(self, graph: KnowledgeGraph, db_path: str | None = None) -> list[AnomalyResult]:
        """Run all anomaly detection checks and persist results."""
        db = db_path or self._db_path
        all_results: list[AnomalyResult] = []
        all_results.extend(self.detect_low_confidence_entities(graph))
        all_results.extend(self.detect_isolated_entities(graph))
        all_results.extend(self.detect_memory_decay(db))
        all_results.extend(self.detect_card_backlog(db))
        for result in all_results:
            self.record(result)
        return all_results
