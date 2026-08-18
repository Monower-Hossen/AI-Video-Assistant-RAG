"""
core/extractor.py
────────────────────────────────────────────────────────────────────────────
Extracts action items, key decisions, and open questions from the meeting
transcript using Mistral AI via LangChain.

Requires MISTRAL_API_KEY in your .env file.
────────────────────────────────────────────────────────────────────────────
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm import get_llm

def _run_extraction(transcript: str, instruction: str) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"{instruction}\n\n"
            "Respond in English as a short bullet list (using '- ' prefixes). "
            "If none are found, respond with exactly: 'None found.'",
        ),
        ("human", "Meeting transcript:\n{transcript}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"transcript": transcript}).strip()


def extract_action_items(transcript: str) -> str:
    return _run_extraction(
        transcript,
        "Extract all clear action items / tasks assigned during this "
        "meeting, including the responsible person if mentioned.",
    )


def extract_key_decisions(transcript: str) -> str:
    return _run_extraction(
        transcript,
        "Extract all key decisions that were made during this meeting.",
    )


def extract_questions(transcript: str) -> str:
    return _run_extraction(
        transcript,
        "Extract all open or unresolved questions raised during this meeting.",
    )
