"""KnowledgeGraph: Kuzu graph database for entity-relation storage.

Property graph with Cypher queries, scalable to 100K+ nodes.
Node label: Entity
Relation label: RELATES_TO
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Optional

import kuzu

from src.models.entity import Entity, EntityType
from src.models.relation import Relation, RelationType

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Manages entities and relations in an embedded Kuzu database."""

    def __init__(self, db_path: str) -> None:
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db_path = db_path
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create Kuzu schema if tables do not already exist."""
        try:
            self._conn.execute(
                "CREATE NODE TABLE Entity("
                "id STRING, name STRING, type STRING, aliases STRING, properties STRING, "
                "confidence DOUBLE, source_note_ids STRING, created STRING, modified STRING, "
                "PRIMARY KEY (id))"
            )
        except Exception:
            pass

        try:
            self._conn.execute(
                "CREATE REL TABLE RELATES_TO(FROM Entity TO Entity, "
                "type STRING, confidence DOUBLE, properties STRING, created STRING)"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _entity_to_dict(self, entity: Entity) -> dict[str, Any]:
        return {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type.value,
            "aliases": json.dumps(entity.aliases),
            "properties": json.dumps(entity.properties),
            "confidence": entity.confidence,
            "source_note_ids": json.dumps(entity.source_note_ids),
        }

    def _dict_to_entity(self, data: dict[str, Any]) -> Entity:
        return Entity(
            id=data["id"],
            name=data["name"],
            type=EntityType(data["type"]),
            aliases=json.loads(data["aliases"]),
            properties=json.loads(data["properties"]),
            confidence=float(data["confidence"]),
            source_note_ids=json.loads(data["source_note_ids"]),
        )

    def _row_to_entity(self, row: tuple[Any, ...] | list[Any]) -> Entity:
        keys = ["id", "name", "type", "aliases", "properties", "confidence", "source_note_ids"]
        return self._dict_to_entity(dict(zip(keys, row)))

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def create_entity(self, entity: Entity) -> Entity:
        """Add or replace an entity in Kuzu."""
        now = datetime.utcnow().isoformat()
        data = self._entity_to_dict(entity)
        data["created"] = now
        data["modified"] = now
        self._conn.execute(
            "MERGE (a:Entity {"
            "id: $id, name: $name, type: $type, aliases: $aliases, properties: $properties, "
            "confidence: $confidence, source_note_ids: $source_note_ids, created: $created, modified: $modified})",
            data,
        )
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by its UUID."""
        result = self._conn.execute(
            "MATCH (a:Entity) WHERE a.id = $id "
            "RETURN a.id, a.name, a.type, a.aliases, a.properties, a.confidence, a.source_note_ids",
            {"id": entity_id},
        )
        rows = result.get_all()
        if not rows:
            return None
        return self._row_to_entity(rows[0])

    def update_entity(self, entity: Entity) -> Entity:
        """Update an existing entity's properties."""
        now = datetime.utcnow().isoformat()
        data = self._entity_to_dict(entity)
        self._conn.execute(
            "MATCH (a:Entity) WHERE a.id = $id SET "
            "a.name = $name, a.type = $type, a.aliases = $aliases, a.properties = $properties, "
            "a.confidence = $confidence, a.source_note_ids = $source_note_ids, a.modified = $modified",
            {**data, "modified": now},
        )
        return entity

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relations."""
        check = self._conn.execute(
            "MATCH (a:Entity) WHERE a.id = $id RETURN a.id",
            {"id": entity_id},
        )
        rows = check.get_all()
        if not rows:
            return False
        self._conn.execute(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.id = $id DELETE r",
            {"id": entity_id},
        )
        self._conn.execute(
            "MATCH (a:Entity)<-[r:RELATES_TO]-(b:Entity) WHERE a.id = $id DELETE r",
            {"id": entity_id},
        )
        self._conn.execute(
            "MATCH (a:Entity) WHERE a.id = $id DELETE a",
            {"id": entity_id},
        )
        return True

    def find_entities(
        self, type: str | None = None, name: str | None = None, limit: int = 20
    ) -> list[Entity]:
        """Search/filter entities."""
        if type and name:
            query = (
                "MATCH (a:Entity) WHERE a.type = $type AND a.name CONTAINS $name "
                "RETURN a.id, a.name, a.type, a.aliases, a.properties, a.confidence, a.source_note_ids "
                "ORDER BY a.modified DESC LIMIT $limit"
            )
            params = {"type": type, "name": name, "limit": limit}
        elif type:
            query = (
                "MATCH (a:Entity) WHERE a.type = $type "
                "RETURN a.id, a.name, a.type, a.aliases, a.properties, a.confidence, a.source_note_ids "
                "ORDER BY a.modified DESC LIMIT $limit"
            )
            params = {"type": type, "limit": limit}
        elif name:
            query = (
                "MATCH (a:Entity) WHERE a.name CONTAINS $name "
                "RETURN a.id, a.name, a.type, a.aliases, a.properties, a.confidence, a.source_note_ids "
                "ORDER BY a.modified DESC LIMIT $limit"
            )
            params = {"name": name, "limit": limit}
        else:
            query = (
                "MATCH (a:Entity) "
                "RETURN a.id, a.name, a.type, a.aliases, a.properties, a.confidence, a.source_note_ids "
                "ORDER BY a.modified DESC LIMIT $limit"
            )
            params = {"limit": limit}
        result = self._conn.execute(query, params)
        rows = result.get_all()
        return [self._row_to_entity(row) for row in rows]

    def find_entities_by_alias(self, alias: str, limit: int = 20) -> list[Entity]:
        """Search entities by alias substring.

        ``aliases`` is stored as a JSON-encoded string, so a ``CONTAINS`` match
        on the serialized text effectively performs a substring search across all
        alias values of each entity. This mirrors ``find_entities(name=...)`` but
        matches against the aliases column instead of the name column.

        Args:
            alias: Substring to search for within the aliases JSON string.
            limit: Maximum number of entities to return.

        Returns:
            A list of matching Entity objects, ordered by most-recently-modified.
        """
        result = self._conn.execute(
            "MATCH (a:Entity) WHERE a.aliases CONTAINS $alias "
            "RETURN a.id, a.name, a.type, a.aliases, a.properties, "
            "a.confidence, a.source_note_ids "
            "ORDER BY a.modified DESC LIMIT $limit",
            {"alias": alias, "limit": limit},
        )
        rows = result.get_all()
        return [self._row_to_entity(row) for row in rows]

    def find_entities_by_note_id(self, note_id: str) -> list[Entity]:
        """Return every entity whose ``source_note_ids`` contains ``note_id``.

        ``source_note_ids`` is stored as a JSON string, so this matches on the
        serialized text. ``note_id`` is a UUID, so a substring CONTAINS match is
        exact in practice (no false positives) and avoids the bounded full scan
        used elsewhere.
        """
        result = self._conn.execute(
            "MATCH (a:Entity) WHERE a.source_note_ids CONTAINS $note_id "
            "RETURN a.id, a.name, a.type, a.aliases, a.properties, a.confidence, a.source_note_ids",
            {"note_id": note_id},
        )
        rows = result.get_all()
        return [self._row_to_entity(row) for row in rows]

    def count_entities(self) -> int:
        result = self._conn.execute("MATCH (a:Entity) RETURN COUNT(a)")
        rows = result.get_all()
        return int(rows[0][0]) if rows else 0

    def random_entity(self) -> Optional[Entity]:
        """Return a random entity from the graph."""
        count = self.count_entities()
        if count == 0:
            return None
        offset = random.randint(0, count - 1)
        result = self._conn.execute(
            "MATCH (a:Entity) "
            "RETURN a.id, a.name, a.type, a.aliases, a.properties, a.confidence, a.source_note_ids "
            "SKIP $offset LIMIT 1",
            {"offset": offset},
        )
        rows = result.get_all()
        if not rows:
            return None
        return self._row_to_entity(rows[0])

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    def create_relation(self, relation: Relation) -> Relation:
        """Add a relation between two existing entities."""
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            "MATCH (a:Entity), (b:Entity) WHERE a.id = $sid AND b.id = $tid "
            "CREATE (a)-[:RELATES_TO {type: $type, confidence: $confidence, properties: $properties, created: $created}]->(b)",
            {
                "sid": relation.source_entity_id,
                "tid": relation.target_entity_id,
                "type": relation.type.value,
                "confidence": relation.confidence,
                "properties": json.dumps(relation.properties),
                "created": now,
            },
        )
        return relation

    def get_relations(self, entity_id: str) -> list[Relation]:
        """Get all relations (both incoming and outgoing) for an entity."""
        relations: list[Relation] = []
        # Outgoing
        result = self._conn.execute(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.id = $id "
            "RETURN a.id, b.id, r.type, r.confidence, r.properties",
            {"id": entity_id},
        )
        for row in result.get_all():
            relations.append(
                Relation(
                    id="",
                    source_entity_id=row[0],
                    target_entity_id=row[1],
                    type=RelationType(row[2]),
                    confidence=float(row[3]),
                    properties=json.loads(row[4]),
                )
            )
        # Incoming
        result = self._conn.execute(
            "MATCH (a:Entity)<-[r:RELATES_TO]-(b:Entity) WHERE a.id = $id "
            "RETURN b.id, a.id, r.type, r.confidence, r.properties",
            {"id": entity_id},
        )
        for row in result.get_all():
            relations.append(
                Relation(
                    id="",
                    source_entity_id=row[0],
                    target_entity_id=row[1],
                    type=RelationType(row[2]),
                    confidence=float(row[3]),
                    properties=json.loads(row[4]),
                )
            )
        return relations

    def get_all_relations(self, limit: int = 500) -> list[Relation]:
        """Get all relations across the graph, up to `limit` rows."""
        result = self._conn.execute(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
            "RETURN a.id, b.id, r.type, r.confidence, r.properties LIMIT $limit",
            {"limit": limit},
        )
        relations: list[Relation] = []
        for row in result.get_all():
            relations.append(
                Relation(
                    source_entity_id=row[0],
                    target_entity_id=row[1],
                    type=RelationType(row[2]),
                    confidence=float(row[3]),
                    properties=json.loads(row[4]),
                )
            )
        return relations

    def count_relations(self) -> int:
        result = self._conn.execute("MATCH ()-[r:RELATES_TO]->() RETURN COUNT(r)")
        rows = result.get_all()
        return int(rows[0][0]) if rows else 0

    def find_relation(
        self,
        source_id: str,
        target_id: str,
        type: str,
        predicate: str | None = None,
    ) -> Relation | None:
        """Find a directed relation source->target of ``type`` (optionally matching
        the verbatim ``properties['predicate']``). Returns the first match or None."""
        for rel in self.get_relations(source_id):
            if (
                rel.source_entity_id == source_id
                and rel.target_entity_id == target_id
                and rel.type.value == type
                and (predicate is None or rel.properties.get("predicate") == predicate)
            ):
                return rel
        return None

    def delete_relation(
        self,
        source_id: str,
        target_id: str,
        type: str,
        predicate: str | None = None,
    ) -> int:
        """Delete directed relation(s) source->target of the given ``type``.

        Kuzu stores no per-edge id, so a single edge cannot be addressed directly:
        ``DELETE r`` removes every source->target edge of this type. When
        ``predicate`` is given we delete the whole set, then re-create the edges
        whose ``properties['predicate']`` differs, so only the targeted predicate
        is removed. Returns the number of edges actually removed.
        """
        existing = [
            rel
            for rel in self.get_relations(source_id)
            if rel.source_entity_id == source_id
            and rel.target_entity_id == target_id
            and rel.type.value == type
        ]
        if not existing:
            return 0
        self._conn.execute(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
            "WHERE a.id = $sid AND b.id = $tid AND r.type = $type DELETE r",
            {"sid": source_id, "tid": target_id, "type": type},
        )
        if predicate is None:
            return len(existing)
        survivors = [rel for rel in existing if rel.properties.get("predicate") != predicate]
        for rel in survivors:
            self.create_relation(rel)
        return len(existing) - len(survivors)

    def delete_cooccurrence_relation(self, source_id: str, target_id: str) -> int:
        """Delete co-occurrence ``related_to`` edge(s) source->target.

        Co-occurrence edges are note-derived and carry no ``properties['predicate']``;
        memory-derived ``related_to`` edges always carry one. Kuzu stores no per-edge
        id, so a single edge cannot be addressed: we delete every source->target
        ``related_to`` edge, then re-create only the predicate-bearing (memory-derived)
        ones. Also clears duplicate co-occurrence edges, which the creation path does
        not dedup. Returns the number of co-occurrence edges removed.
        """
        rel_type = RelationType.related_to.value
        existing = [
            rel
            for rel in self.get_relations(source_id)
            if rel.source_entity_id == source_id
            and rel.target_entity_id == target_id
            and rel.type.value == rel_type
        ]
        survivors = [rel for rel in existing if rel.properties.get("predicate") is not None]
        removed = len(existing) - len(survivors)
        if removed == 0:
            return 0
        self._conn.execute(
            "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
            "WHERE a.id = $sid AND b.id = $tid AND r.type = $type DELETE r",
            {"sid": source_id, "tid": target_id, "type": rel_type},
        )
        for rel in survivors:
            self.create_relation(rel)
        return removed

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def find_paths(self, source_id: str, target_id: str, max_depth: int = 3) -> list[list[Entity]]:
        """Find all simple paths from source to target up to max_depth hops."""
        if source_id == target_id:
            entity = self.get_entity(source_id)
            return [[entity]] if entity else []

        # Kuzu does not support parameterised path lengths, so max_depth is interpolated
        result = self._conn.execute(
            f"MATCH p = (a:Entity)-[:RELATES_TO*1..{max_depth}]->(b:Entity) "
            "WHERE a.id = $source_id AND b.id = $target_id "
            "RETURN nodes(p)",
            {"source_id": source_id, "target_id": target_id},
        )
        paths: list[list[Entity]] = []
        for row in result.get_all():
            nodes = row[0]
            path_entities = [self._dict_to_entity(node) for node in nodes]
            paths.append(path_entities)
        return paths

    # ------------------------------------------------------------------
    # Merge entities
    # ------------------------------------------------------------------

    def merge_entities(self, entity_ids: list[str], merged_name: str, merged_type: str) -> Entity:
        """Merge multiple entities into one.

        The first entity_id becomes the canonical entity. All properties,
        aliases, source_note_ids and relations of the other entities are
        merged into it. The other entities are then deleted.
        """
        if not entity_ids:
            raise ValueError("entity_ids must not be empty")

        canonical_id = entity_ids[0]
        canonical = self.get_entity(canonical_id)
        if canonical is None:
            raise ValueError(f"Canonical entity {canonical_id} not found")

        all_aliases = set(canonical.aliases)
        all_source_note_ids = set(canonical.source_note_ids)
        all_source_memory_ids = set(canonical.properties.get("source_memory_ids", []))
        all_properties = dict(canonical.properties)
        max_confidence = canonical.confidence

        for other_id in entity_ids[1:]:
            other = self.get_entity(other_id)
            if other is None:
                continue
            all_aliases.update(other.aliases)
            all_source_note_ids.update(other.source_note_ids)
            all_source_memory_ids.update(other.properties.get("source_memory_ids", []))
            # dict.update would overwrite list-valued provenance; source_memory_ids
            # is unioned back in below.
            all_properties.update(other.properties)
            if other.confidence > max_confidence:
                max_confidence = other.confidence

            # Redirect outgoing relations of other to canonical
            outgoing = self._conn.execute(
                "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.id = $id "
                "RETURN b.id, r.type, r.confidence, r.properties",
                {"id": other_id},
            )
            for row in outgoing.get_all():
                self._conn.execute(
                    "MATCH (a:Entity), (b:Entity) WHERE a.id = $canonical_id AND b.id = $bid "
                    "CREATE (a)-[:RELATES_TO {type: $type, confidence: $confidence, properties: $properties}]->(b)",
                    {
                        "canonical_id": canonical_id,
                        "bid": row[0],
                        "type": row[1],
                        "confidence": float(row[2]),
                        "properties": str(row[3]),
                    },
                )

            # Redirect incoming relations of other to canonical
            incoming = self._conn.execute(
                "MATCH (a:Entity)<-[r:RELATES_TO]-(b:Entity) WHERE a.id = $id "
                "RETURN b.id, r.type, r.confidence, r.properties",
                {"id": other_id},
            )
            for row in incoming.get_all():
                self._conn.execute(
                    "MATCH (a:Entity), (b:Entity) WHERE a.id = $canonical_id AND b.id = $bid "
                    "CREATE (b)-[:RELATES_TO {type: $type, confidence: $confidence, properties: $properties}]->(a)",
                    {
                        "canonical_id": canonical_id,
                        "bid": row[0],
                        "type": row[1],
                        "confidence": float(row[2]),
                        "properties": str(row[3]),
                    },
                )

            # Delete other entity (and its remaining relations)
            self.delete_entity(other_id)

        # Update canonical entity
        canonical.name = merged_name
        canonical.type = EntityType(merged_type)
        canonical.aliases = list(all_aliases)
        canonical.source_note_ids = list(all_source_note_ids)
        if all_source_memory_ids:
            all_properties["source_memory_ids"] = list(all_source_memory_ids)
        canonical.properties = all_properties
        canonical.confidence = max_confidence
        self.update_entity(canonical)
        return canonical

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_entity_types(self) -> dict[str, int]:
        result = self._conn.execute("MATCH (a:Entity) RETURN a.type, COUNT(a)")
        types: dict[str, int] = {}
        for row in result.get_all():
            types[row[0]] = int(row[1])
        return types
