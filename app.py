"""Gradio entrypoint for the music chat recommender.

Run locally:        python app.py
Run on HF Spaces:   the Space SDK calls `python app.py` automatically.
                    Secrets (GROQ_API_KEY, GEMINI_API_KEY, YOUTUBE_API_KEY)
                    must be set in the Space's "Settings → Variables and
                    secrets" panel.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from recommender.agent import RecommenderAgent
from recommender.conversation import ConversationState
from recommender.render import render_error, render_response

load_dotenv()

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


# --------------------------------------------------------------------------- #
#  Lazy global agent.                                                         #
#                                                                             #
#  Why global: the API clients hold an in-memory token / LRU cache. Per-user  #
#  state lives in `gr.State`, not in the agent.                               #
# --------------------------------------------------------------------------- #


_AGENT: RecommenderAgent | None = None
_AGENT_INIT_ERROR: str | None = None


def get_agent() -> RecommenderAgent | None:
    global _AGENT, _AGENT_INIT_ERROR
    if _AGENT is None and _AGENT_INIT_ERROR is None:
        try:
            _AGENT = RecommenderAgent()
        except Exception as e:  # noqa: BLE001
            _AGENT_INIT_ERROR = str(e)
    return _AGENT


def missing_keys() -> list[str]:
    """The agent needs at least one LLM key (Groq OR Gemini). YouTube is
    optional — the YouTube client degrades to a 'search on YouTube' link
    when absent. iTunes Search API needs no key at all."""
    if os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return []
    return ["GROQ_API_KEY (or GEMINI_API_KEY)"]


# --------------------------------------------------------------------------- #
#  Chat callback.                                                             #
# --------------------------------------------------------------------------- #


def respond(message: str, history: list[dict], state: ConversationState | None):
    """Single-turn handler. Returns (assistant_html_string, updated_state)."""
    if state is None:
        state = ConversationState()

    if not message or not message.strip():
        return render_error("Please type a request, e.g. 'recommend me sad nighttime songs'."), state

    keys = missing_keys()
    if keys:
        return (
            render_error(
                "Missing API keys: "
                + ", ".join(keys)
                + ". Add them to .env (locally) or to the Space's Secrets panel."
            ),
            state,
        )

    agent = get_agent()
    if agent is None:
        return render_error(f"Recommender failed to start: {_AGENT_INIT_ERROR}"), state

    try:
        resp = agent.respond(message, state)
    except Exception as e:  # noqa: BLE001
        return render_error(f"Internal error: {e}"), state

    return render_response(resp), state


# --------------------------------------------------------------------------- #
#  UI helpers                                                                 #
# --------------------------------------------------------------------------- #


def _embed_image(path: Path) -> str | None:
    """Inline an image as a data URL so we don't need a static file server."""
    if not path.exists():
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _hero_html() -> str:
    cover = _embed_image(ASSETS_DIR / "cover.png")
    icon = _embed_image(ASSETS_DIR / "icon.png")
    bg = (
        f'background-image: linear-gradient(180deg, rgba(30,27,75,0.20) 0%, rgba(219,39,119,0.10) 100%), '
        f'url("{cover}");'
        f'background-size: cover; background-position: center;'
    ) if cover else (
        "background: linear-gradient(135deg, #4338ca 0%, #ec4899 100%);"
    )
    icon_html = (
        f'<img src="{icon}" alt="" style="width:42px;height:42px;border-radius:10px;'
        f'box-shadow:0 4px 14px rgba(0,0,0,0.25);" />'
        if icon else "🎧"
    )
    return f"""
    <div class="mcr-hero" style="{bg}">
      <div class="mcr-hero-inner">
        <div class="mcr-hero-badge">{icon_html}</div>
        <h1 class="mcr-hero-title">Music Chat Recommender</h1>
        <p class="mcr-hero-subtitle">
          Tell me what you want to hear &mdash; in plain English. I'll search across
          millions of tracks, listen to your follow-ups, and play them back here.
        </p>
        <div class="mcr-hero-pills">
          <span class="mcr-pill">🧠 Groq gpt-oss-120b</span>
          <span class="mcr-pill">🎵 iTunes Search</span>
          <span class="mcr-pill">▶ YouTube Embed</span>
          <span class="mcr-pill">🔁 Multi-turn memory</span>
        </div>
      </div>
    </div>
    """


def _about_html() -> str:
    return """
    <details class="mcr-about">
      <summary>How does this work?</summary>
      <div class="mcr-about-body">
        <ol>
          <li><b>Intent parsing.</b> Your message + the conversation history go to a large
          language model (Groq's gpt-oss-120b by default), which returns a
          structured <code>SearchQuery</code> JSON: moods, genres, seed artists,
          year range, exclusions, "diversify" hint, etc.</li>
          <li><b>Catalog search.</b> The structured query is fanned out into 1-6
          targeted iTunes Search API calls. Results are merged, de-duplicated,
          year-filtered, and have your blacklist removed.</li>
          <li><b>Rerank &amp; reply.</b> The candidate pool goes back to the LLM
          along with a <code>target_count</code>. It picks the best ordered
          subset and writes a short conversational reply.</li>
          <li><b>Playback.</b> For each chosen track, a YouTube Data API call
          finds an embeddable <code>videoId</code>; the iTunes 30s preview is
          attached as an HTML5 audio control.</li>
        </ol>
        <p>Per-session memory is held in <code>gr.State</code> &mdash; once you reload
        the page or hit <i>Reset conversation</i>, your exclusions and history reset.</p>
        <p style="opacity:0.75;font-size:13px;">Source code &amp; technical write-up:
        <a href="https://github.com/cedricshan/music-chat-recommender" target="_blank" rel="noopener">GitHub</a>.</p>
      </div>
    </details>
    """


def _footer_html() -> str:
    return """
    <div class="mcr-footer">
      <span>Made with Gradio · Hugging Face Spaces · Groq · iTunes Search · YouTube Data API</span>
      <span class="mcr-footer-sep">·</span>
      <a href="https://github.com/cedricshan/music-chat-recommender" target="_blank" rel="noopener">GitHub</a>
    </div>
    """


# --------------------------------------------------------------------------- #
#  Theme + custom CSS                                                         #
# --------------------------------------------------------------------------- #


CUSTOM_CSS = """
:root {
  --mcr-grad-from: #4338ca;
  --mcr-grad-to:   #ec4899;
}

.mcr-hero {
  position: relative;
  border-radius: 18px;
  margin: 0 0 18px 0;
  padding: 0;
  min-height: 260px;
  overflow: hidden;
  isolation: isolate;
  box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}
.mcr-hero::after {
  content: "";
  position: absolute; inset: 0;
  background: linear-gradient(180deg, rgba(15,15,30,0.0) 30%, rgba(15,15,30,0.55) 100%);
  pointer-events: none;
}
.mcr-hero-inner {
  position: relative;
  z-index: 2;
  padding: 36px 32px 32px 32px;
  color: #ffffff;
  text-shadow: 0 1px 6px rgba(0,0,0,0.35);
}
.mcr-hero-badge {
  display: inline-flex;
  align-items: center; justify-content: center;
  width: 46px; height: 46px;
  margin-bottom: 14px;
  font-size: 28px;
}
.mcr-hero-title {
  margin: 0 0 6px 0;
  font-size: 30px;
  letter-spacing: -0.01em;
  font-weight: 700;
  color: #ffffff;
}
.mcr-hero-subtitle {
  margin: 0 0 14px 0;
  font-size: 15px;
  max-width: 640px;
  color: rgba(255,255,255,0.88);
  line-height: 1.5;
}
.mcr-hero-pills {
  display: flex; flex-wrap: wrap; gap: 6px;
}
.mcr-pill {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 10px;
  font-size: 12px; font-weight: 500;
  border-radius: 999px;
  background: rgba(255,255,255,0.18);
  color: #ffffff;
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  border: 1px solid rgba(255,255,255,0.25);
}

.mcr-about {
  margin: 8px 0 18px 0;
  padding: 14px 18px;
  border-radius: 12px;
  background: var(--background-fill-secondary, rgba(128,128,128,0.06));
  border: 1px solid var(--border-color-primary, rgba(128,128,128,0.18));
}
.mcr-about > summary {
  cursor: pointer; font-weight: 600; font-size: 14px;
  list-style: none; user-select: none;
}
.mcr-about > summary::-webkit-details-marker { display: none; }
.mcr-about > summary::before { content: "▸ "; opacity: 0.6; }
.mcr-about[open] > summary::before { content: "▾ "; opacity: 0.6; }
.mcr-about-body { margin-top: 10px; font-size: 14px; line-height: 1.55; }
.mcr-about-body ol { padding-left: 18px; }
.mcr-about-body li { margin-bottom: 6px; }
.mcr-about-body code {
  background: rgba(128,128,128,0.15); padding: 1px 5px; border-radius: 4px;
  font-size: 12.5px;
}

.mcr-footer {
  margin: 18px 0 4px 0;
  padding-top: 14px;
  font-size: 12px; opacity: 0.6;
  text-align: center;
  border-top: 1px solid var(--border-color-primary, rgba(128,128,128,0.18));
}
.mcr-footer .mcr-footer-sep { margin: 0 8px; }
.mcr-footer a { color: inherit; text-decoration: underline; }

/* Bigger, friendlier example chips */
.examples-holder .gradio-button, .examples .gradio-button {
  border-radius: 999px !important;
}

/* Slightly tighter chat container */
.chatbot { border-radius: 14px !important; }

/* Banner shown when API keys are missing */
.mcr-banner {
  margin: 0 0 14px 0;
  padding: 12px 16px;
  border-radius: 10px;
  background: linear-gradient(90deg, rgba(239,68,68,0.10), rgba(249,115,22,0.10));
  border: 1px solid rgba(239,68,68,0.35);
  font-size: 14px;
}
"""


EXAMPLES = [
    "Recommend me some sad nighttime songs.",
    "Songs for a long road trip through California.",
    "Mix it up — give me something more diverse next time.",
    "Find me upbeat indie tracks from 2018-2022.",
    "Stop recommending Taylor Swift, please.",
    "Anything similar to Bon Iver but more electronic.",
    "Five tracks for a focused study session, no lyrics.",
    "Show me classic 70s rock anthems.",
]


def build_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )


def build_app() -> gr.Blocks:
    icon_path = ASSETS_DIR / "icon.png"
    chatbot = gr.Chatbot(
        height=560,
        sanitize_html=False,
        label="Recommendations",
        avatar_images=(None, str(icon_path) if icon_path.exists() else None),
        show_label=False,
        layout="bubble",
    )

    with gr.Blocks(title="Music Chat Recommender", fill_height=True) as demo:
        gr.HTML(_hero_html())

        keys = missing_keys()
        if keys:
            gr.HTML(
                f'<div class="mcr-banner">⚠️ <b>Missing API keys:</b> '
                f'<code>{", ".join(keys)}</code>. '
                "Add them to <code>.env</code> locally, or in the Space's "
                "<i>Settings → Secrets</i>.</div>"
            )

        gr.HTML(_about_html())

        # Per-session conversation memory.
        state = gr.State(ConversationState())

        # ChatInterface needs nested lists for examples when there are
        # additional_inputs; we pass the user text and let the state default.
        examples_nested = [[ex, None] for ex in EXAMPLES]

        chat = gr.ChatInterface(
            fn=respond,
            chatbot=chatbot,
            additional_inputs=[state],
            additional_outputs=[state],
            examples=examples_nested,
            cache_examples=False,
            title=None,
            description=None,
            fill_height=True,
            save_history=False,
            submit_btn=True,
            stop_btn=False,
            textbox=gr.Textbox(
                placeholder="Tell me what you want to hear…",
                container=False,
                scale=7,
                lines=1,
                max_lines=3,
            ),
        )

        with gr.Row():
            reset_btn = gr.Button("🗘 Reset conversation", variant="secondary", scale=0)
            status = gr.Markdown(visible=False)

        gr.HTML(_footer_html())

        def _reset():
            return ConversationState(), gr.update(value="Conversation reset.", visible=True)

        reset_btn.click(
            _reset,
            inputs=None,
            outputs=[state, status],
        ).then(
            lambda: [],
            inputs=None,
            outputs=chat.chatbot,
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        theme=build_theme(),
        css=CUSTOM_CSS,
    )
