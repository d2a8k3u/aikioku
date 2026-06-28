"""QuestionGenerator: generate review questions from a note."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.note import Note


@dataclass
class GeneratedQuestion:
    """A review question generated from a note.

    Attributes:
        type: Question style — cloze_deletion, qa, or connection.
        question: The question or prompt text.
        answer: The expected answer.
    """

    type: str
    question: str
    answer: str


class QuestionGenerator:
    """Generates review questions from note content using rule-based heuristics."""

    def generate_from_note(self, note: Note, count: int = 5) -> list[GeneratedQuestion]:
        """Generate up to `count` questions from the given note.

        Supports cloze_deletion, qa, and connection types.

        Args:
            note: The source note.
            count: Maximum number of questions to generate.

        Returns:
            A list of GeneratedQuestion objects.
        """
        questions: list[GeneratedQuestion] = []
        sentences = self._split_sentences(note.content)
        keywords = self._extract_keywords(note.content)

        # Cloze deletions from sentences
        for sentence in sentences:
            if len(questions) >= count:
                break
            cloze = self._make_cloze(sentence, keywords, note.title)
            if cloze:
                questions.append(cloze)

        # QA questions
        for sentence in sentences:
            if len(questions) >= count:
                break
            qa = self._make_qa(sentence, note.title)
            if qa:
                questions.append(qa)

        # Connection questions
        if len(keywords) >= 2 and len(questions) < count:
            conn = self._make_connection(keywords, note.title)
            if conn:
                questions.append(conn)

        # Fallback for short notes
        if not questions:
            questions.append(
                GeneratedQuestion(
                    type="qa",
                    question=f"What is the main topic of '{note.title}'?",
                    answer=note.content.strip() or note.title,
                )
            )

        return questions[:count]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract potential keywords from text (capitalized words or quoted terms)."""
        # Find capitalized words longer than 3 chars
        found = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text)
        # Deduplicate preserving order
        seen: set[str] = set()
        result: list[str] = []
        for w in found:
            lw = w.lower()
            if lw not in seen:
                seen.add(lw)
                result.append(w)
        return result

    @staticmethod
    def _make_cloze(sentence: str, keywords: list[str], title: str) -> GeneratedQuestion | None:
        """Create a cloze deletion question by blanking out a keyword."""
        words = sentence.split()
        if len(words) < 4:
            return None
        # Pick a keyword present in the sentence
        target = None
        for kw in keywords:
            if kw.lower() in sentence.lower():
                target = kw
                break
        if target is None:
            # Pick longest word as fallback
            target = max(words, key=len)
            if len(target) <= 3:
                return None
        question = sentence.replace(target, f"{{{{c1::{target}}}}}", 1)
        return GeneratedQuestion(
            type="cloze_deletion",
            question=f"From '{title}': {question}",
            answer=target,
        )

    @staticmethod
    def _make_qa(sentence: str, title: str) -> GeneratedQuestion | None:
        """Create a Q-A question from a sentence."""
        words = sentence.split()
        if len(words) < 5:
            return None
        # Simple heuristic: ask "What does X do/support/enable?" using first noun phrase
        # or generic: "What is stated in the following sentence?"
        # Use a more targeted pattern if the sentence defines something
        match = re.match(r"^([A-Z][a-zA-Z\s]+) is (.*?)\.?$", sentence)
        if match:
            subject = match.group(1).strip()
            rest = match.group(2).strip()
            return GeneratedQuestion(
                type="qa",
                question=f"In '{title}', what is {subject}?",
                answer=rest,
            )
        # Generic fallback
        return GeneratedQuestion(
            type="qa",
            question=f"Regarding '{title}', what is stated in: {sentence}?",
            answer=sentence,
        )

    @staticmethod
    def _make_connection(keywords: list[str], title: str) -> GeneratedQuestion | None:
        """Create a connection question linking two concepts."""
        if len(keywords) < 2:
            return None
        return GeneratedQuestion(
            type="connection",
            question=f"How do '{keywords[0]}' and '{keywords[1]}' relate in the context of '{title}'?",
            answer=f"Both concepts are discussed in '{title}' and are connected through the note's content.",
        )
