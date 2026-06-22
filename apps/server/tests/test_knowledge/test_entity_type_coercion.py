"""Tests for entity-type coercion: the LLM emits free-form types that must map
onto the fixed EntityType enum instead of being dropped."""
from __future__ import annotations

from src.knowledge.pipeline import _coerce_entity_type, _parse_entities_from_llm
from src.models.entity import EntityType


def test_exact_enum_match_case_insensitive():
    assert _coerce_entity_type("Person") is EntityType.Person
    assert _coerce_entity_type("concept") is EntityType.Concept
    assert _coerce_entity_type("ORGANIZATION") is EntityType.Organization


def test_synonyms_map_to_enum():
    assert _coerce_entity_type("ProgrammingLanguage") is EntityType.Concept
    assert _coerce_entity_type("Framework") is EntityType.Concept
    assert _coerce_entity_type("Library") is EntityType.Concept
    assert _coerce_entity_type("Company") is EntityType.Organization
    assert _coerce_entity_type("application") is EntityType.Project


def test_unknown_and_empty_default_to_concept():
    assert _coerce_entity_type("Wizardry") is EntityType.Concept
    assert _coerce_entity_type("") is EntityType.Concept
    assert _coerce_entity_type(None) is EntityType.Concept


def test_parse_keeps_entities_with_unknown_types():
    # The LLM frequently ignores the enum and returns "ProgrammingLanguage" etc.
    # Those entities must be kept (coerced), not dropped.
    resp = '[{"name":"Python","type":"ProgrammingLanguage"},{"name":"Guido","type":"Person"}]'
    ents = _parse_entities_from_llm(resp, note_id="n1")
    names = {(e.name, e.type) for e in ents}
    assert ("Python", EntityType.Concept) in names
    assert ("Guido", EntityType.Person) in names


def test_parse_skips_items_without_name():
    resp = '[{"type":"Concept"},{"name":"","type":"Concept"},{"name":"Real","type":"Concept"}]'
    ents = _parse_entities_from_llm(resp, note_id="n1")
    assert [e.name for e in ents] == ["Real"]
