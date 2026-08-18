"""
core/llm.py
────────────────────────────────────────────────────────────────────────────
Centralized Groq LLM client configuration.

This module provides a configured client for the Groq API. Configuration is
handled via environment variables in your .env file.
────────────────────────────────────────────────────────────────────────────
"""

import os
from langchain_groq import ChatGroq


def get_llm(temperature: float = 0.2) -> ChatGroq:
    """
    Returns a configured instance of the ChatGroq model.
    """
    return ChatGroq(
        # Defaults to the specified model if not set in .env.
        # The model name must be one of the models available on Groq.
        model_name=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        groq_api_key=os.getenv("LLM_API_KEY"),
        temperature=temperature,
        max_retries=3,
        timeout=120,
    )