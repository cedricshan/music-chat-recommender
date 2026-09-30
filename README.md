<p align="center">
  <img src="assets/cover.png" alt="Music Chat Recommender" width="100%"/>
</p>

# Music Chat Recommender

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Cedric1218/music-chat-recommender)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A multi-turn LLM chatbot for music recommendation. You type free-form English
("recommend me sad nighttime songs", "no more Taylor Swift",
"mix it up next time") and the bot uses an LLM (Groq gpt-oss-120b,
with Gemini 2.5 Flash Lite as a fallback) to parse your intent, the
iTunes Search API to fetch candidates, and the YouTube Data API to
embed playable videos in the chat. Per-session memory tracks exclusions,
recent genres, and tracks already shown.

| | |
|---|---|
| **Live demo** | https://huggingface.co/spaces/Cedric1218/music-chat-recommender |
| **Report** | [`project.qmd`](project.qmd) → render to `project.html` |
| **Code entrypoint** | [`app.py`](app.py) |
| **Orchestration** | [`recommender/agent.py`](recommender/agent.py) |

## Run it locally

1. Clone and create a virtualenv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` → `.env` and paste in the API keys
   (links are inside the file). The minimum is one LLM key:
   - **`GROQ_API_KEY`** (recommended) — https://console.groq.com/keys
   - **`GEMINI_API_KEY`** (fallback) — https://aistudio.google.com/apikey
   - **`YOUTUBE_API_KEY`** (optional) — https://console.cloud.google.com/
     → enable "YouTube Data API v3" → Credentials → API key
3. Launch the app:
   ```bash
   python app.py
   ```
   Open the printed URL (typically `http://localhost:7860`).

## Reproduce the evaluation

```bash
source .venv/bin/activate
pip install -e ".[analysis]"
python -m analysis.eval        # hits Gemini + Spotify; ~2 minutes
quarto render project.qmd      # produces project.html with figures
```

## Run the unit tests (no API keys needed)

```bash
pip install -e ".[dev]"
pytest tests/
```

## Deploy to Hugging Face Spaces

The YAML frontmatter at the top of this file is the Space configuration.
There are two paths:

**Option A — `gradio deploy` (one command):**
```bash
pip install gradio
gradio deploy           # asks for HF token + Space name on first run
```

**Option B — push the repo to the Space's git remote:**
```bash
huggingface-cli login   # paste your HF write token
git remote add space https://huggingface.co/spaces/<your-username>/music-chat-recommender
git push space main
```

Either way, after the first push, set the secrets in the Space's
*Settings → Variables and secrets* panel
(`GROQ_API_KEY`, optionally `GEMINI_API_KEY` and `YOUTUBE_API_KEY`).
The Space will auto-restart and the chat will go live.

A GitHub Action that mirrors `main` to the Space on every push is
provided in [`.github/workflows/sync-to-space.yml`](.github/workflows/sync-to-space.yml)
— set `HF_TOKEN` and `HF_SPACE` repository secrets to enable it.

## Project layout

```
.
├── app.py                          Gradio entrypoint (HF Space root)
├── recommender/
│   ├── agent.py                    Top-level RecommenderAgent.respond()
│   ├── conversation.py             Per-session memory (exclusions, …)
│   ├── llm.py                      Gemini wrapper + Pydantic schemas
│   ├── prompts.py                  System prompts for both LLM calls
│   ├── spotify_client.py           Client Credentials auth + search
│   ├── youtube_client.py           Data API v3 search → videoId
│   └── render.py                   HTML card rendering
├── analysis/
│   ├── labeled_intents.py          30 hand-labeled benchmark utterances
│   ├── eval.py                     Run all three experiments
│   └── eval_results.json           Cached results consumed by the QMD
├── tests/test_intent_parsing.py    Pure-logic unit tests (no network)
├── project.qmd                     Quarto report
├── pyproject.toml                  Local dev / analysis deps
├── requirements.txt                HF Spaces deps
└── .env.example                    Documents the four required keys
```
