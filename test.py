"""
test.py
────────────────────────────────────────────────────────────────────────────
AI Video Assistant - Environment & Groq Smoke Tests

Usage:
    python test.py
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()


def test_env_key():
    """Check LLM API key."""
    key = os.getenv("LLM_API_KEY")

    assert key and key != "your_api_key_here", (
        "LLM_API_KEY is missing or not set. "
        "Please add your Groq API key to .env"
    )

    print("✅ LLM_API_KEY found in environment.")


def test_imports():
    """Check required Python packages."""
    import whisper  # noqa: F401
    import yt_dlp  # noqa: F401
    from deep_translator import GoogleTranslator  # noqa: F401
    from langchain_groq import ChatGroq  # noqa: F401
    from groq import Groq  # noqa: F401

    print("✅ All core dependencies import successfully.")


def test_groq_models():
    """Check Groq API and list accessible models."""

    from groq import Groq

    api_key = os.getenv("LLM_API_KEY")

    assert api_key, "LLM_API_KEY is missing."

    # IMPORTANT:
    # Pass LLM_API_KEY explicitly to Groq.
    client = Groq(api_key=api_key)

    try:
        models = client.models.list()

        print("\n✅ Groq API connection successful.")
        print("\n📋 Available Groq models:")

        for model in models.data:
            print(f"   - {model.id}")

        return [model.id for model in models.data]

    except Exception as e:
        print("\n❌ Could not retrieve Groq models.")
        print(f"Error: {e}")
        raise


def test_groq_chat():
    """Test an actual LLM request using an accessible model."""

    from groq import Groq

    api_key = os.getenv("LLM_API_KEY")
    client = Groq(api_key=api_key)

    models = client.models.list()
    model_ids = [model.id for model in models.data]

    # Prefer this model if it is available.
    preferred_models = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    ]

    selected_model = None

    for model in preferred_models:
        if model in model_ids:
            selected_model = model
            break

    if not selected_model:
        print("\n⚠️ Preferred models were not found.")
        print("Available models:")
        for model in model_ids:
            print(f"   - {model}")
        return

    print(f"\n🤖 Testing model: {selected_model}")

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Groq API is working!",
            }
        ],
        temperature=0,
    )

    result = response.choices[0].message.content

    print(f"✅ Groq LLM response: {result}")


def test_translation():
    """Test Bangla -> English translation."""

    from deep_translator import GoogleTranslator

    text = GoogleTranslator(
        source="bn",
        target="en"
    ).translate("আমি ভালো আছি")

    assert text, "Bangla -> English translation returned empty text."

    print(
        f"✅ Bangla -> English translation works: "
        f"'আমি ভালো আছি' -> '{text}'"
    )


if __name__ == "__main__":

    print("=" * 60)
    print("Running AI Video Assistant smoke tests...")
    print("=" * 60)

    try:
        test_env_key()
        test_imports()
        test_groq_models()
        test_groq_chat()
        test_translation()

        print("\n" + "=" * 60)
        print("🎉 All smoke tests passed!")
        print("You are ready to run:")
        print("streamlit run app.py")
        print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print("❌ SMOKE TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print("=" * 60)

        sys.exit(1)