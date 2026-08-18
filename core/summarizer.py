"""
core/summarizer.py
────────────────────────────────────────────────────────────────────────────
Generates the meeting title and summary using Mistral AI via LangChain.

Requires MISTRAL_API_KEY in your .env file.
Get a free key at: https://console.mistral.ai/
────────────────────────────────────────────────────────────────────────────
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm import get_llm


def summarize(transcript: str) -> str:
    """Return a concise English summary of the (already-translated) transcript."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting assistant. Summarize the following "
            "meeting transcript clearly and concisely in English, covering "
            "the main topics discussed, in 4-8 sentences.",
        ),
        ("human", "{transcript}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"transcript": transcript}).strip()


def generate_title(transcript: str) -> str:
    """Return a short (max ~8 word) title for the meeting."""
    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate a short, descriptive title (max 8 words) for this "
            "meeting transcript. Return ONLY the title text - no quotes, "
            "no punctuation at the end, no extra commentary.",
        ),
        ("human", "{transcript}"),
    ])
    chain = prompt | llm | StrOutputParser()
    title = chain.invoke({"transcript": transcript[:2000]})
    return title.strip().strip('"').strip("'")
