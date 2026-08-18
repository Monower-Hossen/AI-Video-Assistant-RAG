# 🎬 AI Video Assistant with RAG

Turn any YouTube video or local video/audio file into an intelligent meeting assistant. This tool transcribes, summarizes, extracts key information, and allows you to chat with your media content. It's an end-to-end **Retrieval-Augmented Generation (RAG)** application designed for meeting analysis.

## Description

This project provides a complete pipeline for processing video or audio inputs. It uses **`yt-dlp`** to handle YouTube URLs and **`pydub`** for local files. The audio is transcribed using a local instance of **OpenAI Whisper**, with support for English, Bangla, and Hinglish. The resulting transcript is then processed by **Mistral AI** via LangChain to generate a title, a concise summary, and extract action items, key decisions, and open questions.

Finally, it builds a RAG chain, allowing you to "chat" with the meeting transcript to ask specific questions. The application can be run as a command-line tool (`main.py`) or as a web UI using Streamlit (`app.py`).

## Features

- 🎥 **Flexible input** — YouTube URL, or upload MP4/MKV/AVI/MOV/MP3/WAV/M4A
- 🗣️ **Multi-language Transcription** — Local Whisper for English, Hinglish, Bangla, and Banglish.
- 📝 **Automated Summarization** — Generates a title and a summary of the transcript.
- 🎯 **Information Extraction** — Pulls out action items, key decisions, and open questions.
- 💬 **Conversational RAG** — Chat with the transcript to find specific information.
- 🖥️ **Dual Interface** — Run as a CLI script (`main.py`) or a rich Streamlit web app (`app.py`).
- 🧱 **Modular & Modern** — Built with a clean structure and the latest LangChain Expression Language (LCEL).

## Architecture

```text
YouTube URL / Upload
        ↓
Audio Processing (yt-dlp / pydub)
        ↓
Audio Chunking
        ↓
Whisper Transcription (local, multi-language)
        ↓
Translation to English (for non-English languages)
        ↓
LLM Analysis (Mistral AI via LangChain)
  - Title Generation
  - Summarization
  - Action Item, Decision, Question Extraction
        ↓
RAG Pipeline Setup (Embeddings + Vector Store)
        ↓
Interactive Chat / Final Output
```

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Video/Audio | yt-dlp, pydub, FFmpeg |
| Speech-to-Text | OpenAI Whisper (local) |
| RAG & LLM | LangChain, Mistral AI |
| Embeddings | HuggingFace `sentence-transformers` |
| Vector DB | ChromaDB |
| Translation | deep-translator |
| Language | Python 3.10+ |

## Project Structure

```text
AI-Video-Assistant-RAG/
│
├── app.py                      # Streamlit web UI
├── main.py                     # CLI entry point
├── pipeline.py                 # Main processing orchestrator
├── requirements.txt
├── .env.example                # Copy to .env and fill in your key
├── README.md
│
├── core/
│   ├── extractor.py            # Extracts items, decisions, questions
│   ├── llm.py                  # Centralized LLM client
│   ├── rag_engine.py           # Builds and runs the RAG chain
│   ├── summarizer.py           # Generates title and summary
│   ├── transcriber.py          # Runs Whisper and translation
│   └── vector_store.py         # Handles embeddings and ChromaDB
│
├── utils/
│   └── audio_processor.py      # Downloads and chunks audio
│
└── downloades/                 # Cached audio files (gitignored)
```

## Installation

### 1. Clone and create a virtual environment

**Windows (PowerShell):**
```powershell
git clone https://github.com/Monower-Hossen/AI-Video-Assistant-RAG
cd AI-Video-Assistant-RAG
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
git clone https://github.com/Monower-Hossen/AI-Video-Assistant-RAG
cd AI-Video-Assistant-RAG
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg (system dependency — NOT a pip package)

FFmpeg is required for audio extraction and is installed separately from Python.

**Windows:**
1. Download a build from https://www.gyan.dev/ffmpeg/builds/ (the "essentials" zip is enough).
2. Extract it, e.g. to `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your `PATH` environment variable.
4. Verify: `ffmpeg -version`

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt update && sudo apt install ffmpeg
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Then edit `.env` and set `LLM_API_KEY`. By default this project uses **Groq**, which has a free tier:

1. Go to https://console.groq.com and create an account.
2. Create an API key.
3. Paste it into `.env` as `LLM_API_KEY`.

You can instead point `LLM_BASE_URL` at OpenAI, Together AI, OpenRouter, or any other OpenAI-compatible provider — just update `LLM_BASE_URL` and `LLM_MODEL` accordingly.

## Running the Application

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Example Usage

1. Paste a YouTube URL (or upload a file) and click **Process Video**.
2. Wait for transcription, chunking, and indexing to complete (progress is shown).
3. Type a question like *"What is this video about?"* and click **Ask**.
4. Check the **Sources / Timestamps** section to see exactly where the answer came from.
5. Use the tabs at the bottom to generate a **Summary**, **Chapters**, **Quiz**, or **Flashcards**.

## RAG Workflow (Bangla + English)

**English:** The video's audio is transcribed by Whisper into timestamped text. The transcript is cleaned, split into overlapping chunks, and each chunk is converted into a vector embedding using a sentence-transformers model. These embeddings are stored in a vector database (FAISS/Chroma). When you ask a question, it is also embedded and compared against the stored chunks to find the most relevant ones (retrieval). Those chunks — not the whole transcript — are passed to the LLM along with your question, and the LLM is instructed to answer *only* using that retrieved context, citing timestamps. This is what "Retrieval-Augmented Generation" means: generation grounded in retrieved evidence rather than the model's memory.

**বাংলা:** ভিডিওর অডিও Whisper দিয়ে টাইমস্ট্যাম্পসহ টেক্সটে রূপান্তর করা হয়। এরপর ট্রান্সক্রিপ্ট পরিষ্কার করে ছোট ছোট অংশে (chunk) ভাগ করা হয়, এবং প্রতিটি অংশকে একটি embedding মডেল দিয়ে ভেক্টরে রূপান্তর করে ভেক্টর ডেটাবেসে (FAISS/Chroma) জমা রাখা হয়। আপনি যখন কোনো প্রশ্ন করেন, সেই প্রশ্নটিকেও ভেক্টরে রূপান্তর করে সবচেয়ে প্রাসঙ্গিক অংশগুলো খুঁজে বের করা হয় (retrieval)। শুধু সেই প্রাসঙ্গিক অংশগুলো — পুরো ট্রান্সক্রিপ্ট নয় — LLM-কে দেওয়া হয়, এবং LLM-কে নির্দেশ দেওয়া হয় শুধুমাত্র সেই তথ্যের ভিত্তিতে উত্তর দিতে, সম্ভব হলে টাইমস্ট্যাম্পসহ। এভাবেই উত্তর সবসময় ভিডিওর প্রকৃত বিষয়বস্তুর সাথে সংগতিপূর্ণ থাকে, মডেলের কল্পনাপ্রসূত তথ্যের উপর নির্ভর করে না।

## Troubleshooting

| Problem | Solution |
|---|---|
| `FFmpeg was not found` | Install FFmpeg and ensure it's on your `PATH` (see step 3 above). |
| `No LLM API key found` | Set `LLM_API_KEY` in `.env`. |
| Whisper is very slow | Use a smaller model (`tiny` or `base`) via the dropdown in the UI, or set `WHISPER_MODEL` in `.env`. |
| YouTube download fails | The video may be private, age-restricted, or region-locked; try a different video or upload the file directly. |
| First run is slow | The embedding and Whisper models are downloaded once and cached locally afterward. |
| `Unsupported file type` | Only MP4, MKV, AVI, MOV, MP3, WAV, and M4A are accepted. |
| Answers seem generic | Try a smaller `CHUNK_SIZE` in `.env` for more precise retrieval, or a larger Whisper model for better transcription accuracy. |

## Screenshots

*(Add screenshots of the UI here once you've run the app.)*

```text
screenshots/
├── home.png
├── chat.png
├── summary.png
└── quiz.png
```

## Future Improvements

- Multi-video / playlist support with cross-video search
- Speaker diarization (who said what)
- Export summaries/quizzes/flashcards to PDF or Anki decks
- Streaming token-by-token answers in the UI
- Support for additional languages with automatic translation
- Dockerfile for one-command deployment

## Limitations

- Transcription accuracy depends on audio quality and the chosen Whisper model size.
- Very long videos may need chunked/batched processing and more memory.
- Free-tier LLM APIs may have rate limits.
- Timestamps are approximate when segments are merged into larger chunks.

## License

MIT License — free to use, modify, and distribute. See `LICENSE` for details (add one when publishing to GitHub).
