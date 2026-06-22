"""Tests for QuestionGenerator."""
from __future__ import annotations

import pytest

from src.models.note import Note
from src.reasoning.question_gen import GeneratedQuestion, QuestionGenerator


@pytest.fixture
def sample_note() -> Note:
    return Note(
        title="Python Programming",
        content="Python is a high-level programming language. It supports object-oriented and functional programming.",
        path="python.md",
    )


class TestQuestionGenerator:
    def test_generate_returns_list(self, sample_note):
        qg = QuestionGenerator()
        questions = qg.generate_from_note(sample_note, count=5)
        assert isinstance(questions, list)
        assert len(questions) <= 5
        assert len(questions) > 0

    def test_generated_question_fields(self, sample_note):
        qg = QuestionGenerator()
        questions = qg.generate_from_note(sample_note, count=3)
        for q in questions:
            assert isinstance(q, GeneratedQuestion)
            assert q.type in ("cloze_deletion", "qa", "connection")
            assert isinstance(q.question, str)
            assert isinstance(q.answer, str)
            assert q.question
            assert q.answer

    def test_count_parameter_respected(self, sample_note):
        qg = QuestionGenerator()
        questions = qg.generate_from_note(sample_note, count=2)
        assert len(questions) <= 2

    def test_cloze_deletion_contains_brackets(self, sample_note):
        qg = QuestionGenerator()
        questions = qg.generate_from_note(sample_note, count=5)
        cloze = [q for q in questions if q.type == "cloze_deletion"]
        for q in cloze:
            assert "{{c1::" in q.question or "[...]" in q.question

    def test_qa_has_question_and_answer(self, sample_note):
        qg = QuestionGenerator()
        questions = qg.generate_from_note(sample_note, count=5)
        qa = [q for q in questions if q.type == "qa"]
        for q in qa:
            assert q.question.endswith("?")
            assert q.answer

    def test_connection_links_note_concepts(self, sample_note):
        qg = QuestionGenerator()
        questions = qg.generate_from_note(sample_note, count=5)
        conn = [q for q in questions if q.type == "connection"]
        for q in conn:
            assert "connection" in q.question.lower() or "link" in q.question.lower() or "relate" in q.question.lower()

    def test_note_title_used_in_questions(self, sample_note):
        qg = QuestionGenerator()
        questions = qg.generate_from_note(sample_note, count=5)
        for q in questions:
            assert sample_note.title.lower() in q.question.lower() or sample_note.title.lower() in q.answer.lower()

    def test_short_note_fallback(self):
        note = Note(title="Short", content="Hi.", path="short.md")
        qg = QuestionGenerator()
        questions = qg.generate_from_note(note, count=5)
        assert isinstance(questions, list)
        # Should still generate at least one basic question
        assert len(questions) >= 1
