"""
core/transcriber.py
────────────────────────────────────────────────────────────────────────────
Speech-to-text using local OpenAI Whisper, with automatic translation to
English for non-English inputs so the downstream LLM (Mistral) always gets
clean English text to work with.

Supported `language` values (passed from app.py / main.py):
    - "english"   -> transcribed directly in English
    - "hinglish"  -> transcribed as Hindi, then auto-translated -> English
    - "bangla"    -> transcribed as Bengali, then auto-translated -> English
    - "banglish"  -> same as "bangla" (Bengali written in Latin/mixed script
                      is still spoken Bengali audio, so Whisper is pointed at
                      the Bengali language model)

No API key is required for this file — Whisper runs 100% locally and
deep-translator's default backend (Google) is used for translation.
────────────────────────────────────────────────────────────────────────────
"""

import whisper
from deep_translator import GoogleTranslator

_model = None  # cached Whisper model (loaded once per process)

# Map our human-friendly language names -> Whisper / translator language codes
LANG_CODE_MAP = {
    "english": "en",
    "hinglish": "hi",
    "bangla": "bn",
    "banglish": "bn",
}


def _get_model(model_size: str = "tiny"):
    """Load (and cache) the local Whisper model."""
    global _model
    if _model is None:
        print(f"[transcriber] Loading Whisper model ({model_size})...")
        # Forcing CPU is the most reliable way to avoid CUDA-related memory
        # errors and silent crashes, especially on systems without a
        # dedicated GPU or with incompatible drivers.
        _model = whisper.load_model(model_size, device="cpu")
    return _model


def _translate_to_english(text: str, src_lang: str) -> str:
    """Best-effort translation to English. Falls back to original text on failure."""
    if not text.strip():
        return text
    try:
        translated = GoogleTranslator(source=src_lang, target="en").translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"[transcriber] Translation failed ({src_lang} -> en): {e}")
        return text


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Transcribe a single audio chunk, translating to English if needed."""
    model = _get_model()
    language = (language or "english").lower().strip()
    whisper_lang = None if language == "english" else LANG_CODE_MAP.get(language, None)

    result = model.transcribe(chunk_path, language=whisper_lang)
    text = result.get("text", "").strip()

    if language in ("hinglish", "bangla", "banglish") and text:
        src_lang = LANG_CODE_MAP.get(language, "auto")
        text = _translate_to_english(text, src_lang)

    return text


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Transcribe every audio chunk and stitch the results into one transcript."""
    full_transcript = []
    total = len(chunks)
    for i, chunk_path in enumerate(chunks):
        print(f"[transcriber] Transcribing chunk {i + 1}/{total}...")
        text = transcribe_chunk(chunk_path, language)
        if text:
            full_transcript.append(text)
    return "\n".join(full_transcript).strip()
