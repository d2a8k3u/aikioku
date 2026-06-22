"""Tests for ProgressiveSummarizer."""
import pytest
from unittest.mock import AsyncMock

from src.models.note import Note
from src.augmentation.summarization import ProgressiveSummarizer


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value="mocked response")
    return llm


@pytest.fixture
def sample_note():
    return Note(
        title="Test Note",
        content="Machine learning is a subset of artificial intelligence. "
        "It involves training algorithms on data to make predictions. "
        "Deep learning is a further subset using neural networks. "
        "Applications include image recognition, NLP, and recommendation systems. "
        "The field has grown rapidly due to increased compute power and data availability.",
        path="/notes/test.md",
    )


@pytest.mark.asyncio
async def test_summarize_returns_all_three_levels(mock_llm, sample_note):
    """summarize() should return all 3 levels: brief, detailed, one-liner."""
    mock_llm.complete.side_effect = [
        "- ML is a subset of AI\n- Uses algorithms trained on data\n- Deep learning uses neural networks",
        "Machine learning is a subset of AI that trains algorithms on data to make predictions. Deep learning extends this with neural networks.",
        "Machine learning enables computers to learn from data.",
    ]
    summarizer = ProgressiveSummarizer(llm_provider=mock_llm)
    result = await summarizer.summarize(sample_note)
    assert "brief" in result
    assert "detailed" in result
    assert "one-liner" in result


@pytest.mark.asyncio
async def test_brief_shorter_than_detailed(mock_llm, sample_note):
    """Brief summary should be shorter than detailed summary."""
    mock_llm.complete.side_effect = [
        "- ML is AI subset\n- Trains on data\n- Deep learning uses neural nets",
        "Machine learning is a subset of artificial intelligence that involves training algorithms on data to make predictions. Deep learning is a further subset that uses neural networks for more complex pattern recognition.",
        "ML is a subset of AI.",
    ]
    summarizer = ProgressiveSummarizer(llm_provider=mock_llm)
    result = await summarizer.summarize(sample_note)
    assert len(result["brief"]) < len(result["detailed"])


@pytest.mark.asyncio
async def test_one_liner_is_single_sentence(mock_llm, sample_note):
    """One-liner should be a single sentence (no period in the middle)."""
    mock_llm.complete.side_effect = [
        "- Point one\n- Point two",
        "Detailed paragraph summary here.",
        "Machine learning is a subset of AI that learns from data.",
    ]
    summarizer = ProgressiveSummarizer(llm_provider=mock_llm)
    result = await summarizer.summarize(sample_note)
    one_liner = result["one-liner"].strip()
    # Remove trailing period for counting
    stripped = one_liner.rstrip(".")
    assert "." not in stripped, f"one-liner has multiple sentences: {one_liner}"


def test_prompt_for_brief_asks_for_bullets(mock_llm):
    """_build_prompt for 'brief' should mention bullet points."""
    summarizer = ProgressiveSummarizer(llm_provider=mock_llm)
    prompt = summarizer._build_prompt("Some content here.", "brief")
    assert "bullet" in prompt.lower()


def test_prompt_for_one_liner_asks_for_single_sentence(mock_llm):
    """_build_prompt for 'one-liner' should mention single sentence."""
    summarizer = ProgressiveSummarizer(llm_provider=mock_llm)
    prompt = summarizer._build_prompt("Some content here.", "one-liner")
    assert "single sentence" in prompt.lower()
