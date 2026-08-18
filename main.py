# Add project root to sys.path to allow for absolute imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─── Fix: Prevent potential memory-related crashes with torch ────────────────
try:
    import torch

    # Prevent torch.classes path issues
    torch.classes.__path__ = []  # type: ignore[attr-defined]

    # Fix for CUBLAS_STATUS_NOT_SUPPORTED on some GPU/CUDA setups
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False

except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────


from dotenv import load_dotenv

# Load .env before importing modules that may need API keys
load_dotenv()

from pipeline import run_pipeline
from core.rag_engine import ask_question


def check_api_key():
    """Check whether LLM_API_KEY is available."""

    api_key = os.getenv("LLM_API_KEY")

    if not api_key or api_key == "your_api_key_here":
        print("\n❌ LLM_API_KEY is missing.")
        print("Please add your Groq API key to the .env file.")
        return False

    print("✅ LLM_API_KEY found in environment.")
    return True


if __name__ == "__main__":

    print("=" * 60)
    print("🚀 AI VIDEO ASSISTANT")
    print("=" * 60)

    # Check API key
    import os

    if not check_api_key():
        sys.exit(1)

    # CLI input
    source = input(
        "\nEnter YouTube URL or local file path: "
    ).strip()

    language = (
        input(
            "Language (english/hinglish/bangla/banglish): "
        ).strip().lower()
        or "english"
    )

    print("\n🚀 Starting AI Video Assistant pipeline...")

    try:

        # ─────────────────────────────────────────────────────────────────────
        # Phase 1 — Process video
        # ─────────────────────────────────────────────────────────────────────

        result = run_pipeline(source, language)

        print("\n" + "=" * 60)
        print("📊 VIDEO ANALYSIS RESULT")
        print("=" * 60)

        print(f"\n📌 Title:")
        print(result.video_info.title)

        print(f"\n📋 Summary:")
        print(result.summary)

        print(f"\n✅ Action Items:")
        print(result.action_items)

        print(f"\n🔑 Key Decisions:")
        print(result.key_decisions)

        print(f"\n❓ Open Questions:")
        print(result.open_questions)

        print("=" * 60)

        # ─────────────────────────────────────────────────────────────────────
        # Phase 2 — Chat with video using RAG
        # ─────────────────────────────────────────────────────────────────────

        print("\n💬 Chat with your meeting")
        print("Type 'exit', 'quit', or 'q' to stop.\n")

        rag_chain = result.rag_chain

        while True:

            question = input("You: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            try:

                answer = ask_question(
                    rag_chain,
                    question
                )

                print(f"\n🤖 Assistant: {answer}\n")

            except Exception as e:

                print("\n❌ Error while answering question:")
                print(e)
                print()

    except Exception as e:

        print("\n" + "=" * 60)
        print("❌ PIPELINE ERROR")
        print("=" * 60)

        error_message = str(e).lower()

        if (
            "authenticationerror" in error_message
            or "invalid api key" in error_message
            or "api_key" in error_message
            or "authentication" in error_message
        ):

            print(
                "❌ Authentication Error: "
                "Your LLM API key is missing or invalid."
            )

            print("\nPlease check:")

            print("1. Your `.env` file exists.")
            print("2. `LLM_API_KEY` is set in `.env`.")
            print("3. Your Groq API key is valid.")
            print("4. No extra spaces exist around the API key.")

        elif "model_not_found" in error_message:

            print("❌ Groq model was not found.")

            print(
                "\nYour current Groq account does not provide "
                "the configured model."
            )

            print(
                "\nUse one of your currently available models:"
            )

            print("  - openai/gpt-oss-120b")
            print("  - openai/gpt-oss-20b")
            print("  - qwen/qwen3.6-27b")

            print(
                "\nCheck the ChatGroq configuration in "
                "`pipeline.py` or your LLM module."
            )

        else:

            print(f"An unexpected error occurred: {e}")

            print(
                "\nFor more details, please check "
                "the full traceback above."
            )

        print("=" * 60)