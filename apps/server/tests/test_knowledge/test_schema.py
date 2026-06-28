"""Tests for knowledge schema registry."""

from __future__ import annotations


from src.models.entity import EntityType
from src.models.relation import RelationType


class TestSchemaRegistry:
    def test_import(self):
        pass

    def test_entity_types_seeded(self):
        from src.knowledge.schema import SchemaRegistry

        reg = SchemaRegistry()
        for et in EntityType:
            info = reg.get_type(et.value)
            assert info is not None
            assert info["name"] == et.value

    def test_relation_types_seeded(self):
        from src.knowledge.schema import SchemaRegistry

        reg = SchemaRegistry()
        for rt in RelationType:
            info = reg.get_type(rt.value)
            assert info is not None
            assert info["name"] == rt.value

    def test_register_entity_type(self):
        from src.knowledge.schema import SchemaRegistry

        reg = SchemaRegistry()
        reg.register_entity_type("CustomEntity", "A custom entity for testing")
        info = reg.get_type("CustomEntity")
        assert info == {"name": "CustomEntity", "description": "A custom entity for testing"}

    def test_register_relation_type(self):
        from src.knowledge.schema import SchemaRegistry

        reg = SchemaRegistry()
        reg.register_relation_type("knows", "A knows B", domain="Person", range="Person")
        info = reg.get_type("knows")
        assert info == {
            "name": "knows",
            "description": "A knows B",
            "domain": "Person",
            "range": "Person",
        }

    def test_list_types(self):
        from src.knowledge.schema import SchemaRegistry

        reg = SchemaRegistry()
        reg.register_entity_type("A", "desc A")
        reg.register_entity_type("B", "desc B")
        reg.register_relation_type("rel1", "desc1", "A", "B")
        types = reg.list_types()
        assert "entity_types" in types
        assert "relation_types" in types
        assert "A" in types["entity_types"]
        assert "B" in types["entity_types"]
        assert "rel1" in types["relation_types"]

    def test_get_type_missing(self):
        from src.knowledge.schema import SchemaRegistry

        reg = SchemaRegistry()
        assert reg.get_type("nonexistent") is None
