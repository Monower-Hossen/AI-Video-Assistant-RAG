"""
Main processing pipeline that orchestrates video ingestion, transcription,
and RAG indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain

if TYPE_CHECKING:
    pass


@dataclass
class VideoInfo:
    """A consolidated object with all info about a processed video."""
    title: str

@dataclass
class PipelineResult:
    """The final output of the processing pipeline."""
    video_info: VideoInfo
    transcript: str
    summary: str
    action_items: str
    key_decisions: str
    open_questions: str
    rag_chain: any


def run_pipeline(
    source: str,
    language: str = "english"
) -> PipelineResult:
    """
    Execute the full ingestion, transcription, and indexing pipeline.

    Args:
        source: The YouTube URL or local file path.
        language: The language of the audio.

    Returns:
        A PipelineResult containing the analysis.
    """
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language)

    # Run analysis tasks sequentially to avoid hitting API rate limits
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return PipelineResult(
        video_info=VideoInfo(title=title),
        transcript=transcript,
        summary=summary,
        action_items=action_items,
        key_decisions=decisions,
        open_questions=questions,
        rag_chain=rag_chain,
    )