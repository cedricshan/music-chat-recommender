"""HTML rendering for chat messages.

Gradio Chatbot accepts raw HTML when constructed with `sanitize_html=False`.
We render each recommendation as a self-contained card with album art, the
30s preview (when iTunes provides one), an "Open in Apple Music" button,
and a collapsible YouTube iframe — collapsed by default so the chat stays
scannable.
"""

from __future__ import annotations

import html
import secrets

from recommender.agent import AgentResponse, TrackRecommendation


# --------------------------------------------------------------------------- #
#  CSS injected once per assistant message. Scoped via a unique id so we      #
#  don't fight Gradio's global theme.                                         #
# --------------------------------------------------------------------------- #


_CARD_CSS = """
<style>
.mcr-wrap {
    font-family: inherit;
    --mcr-grad-from: #4338ca;
    --mcr-grad-to:   #ec4899;
}
.mcr-message {
    margin: 2px 0 14px 0;
    line-height: 1.5;
    font-size: 14.5px;
}
.mcr-meta {
    margin: 12px 0 4px 0;
    font-size: 11px;
    opacity: 0.55;
    display: flex; flex-wrap: wrap; gap: 6px;
}
.mcr-meta .mcr-meta-pill {
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(128,128,128,0.10);
}
.mcr-grid {
    display: flex; flex-direction: column; gap: 12px;
}
.mcr-card {
    display: flex; gap: 14px;
    padding: 12px;
    border-radius: 14px;
    border: 1px solid var(--border-color-primary, rgba(128,128,128,0.20));
    background: var(--background-fill-secondary, rgba(255,255,255,0.04));
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}
.mcr-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(67, 56, 202, 0.10);
    border-color: rgba(99, 102, 241, 0.45);
}
.mcr-art {
    flex: 0 0 96px;
    width: 96px; height: 96px;
    border-radius: 10px;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(67, 56, 202, 0.15), rgba(236, 72, 153, 0.15));
    box-shadow: 0 2px 6px rgba(0,0,0,0.10);
    position: relative;
}
.mcr-art img {
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
}
.mcr-body {
    flex: 1 1 auto; min-width: 0;
    display: flex; flex-direction: column; gap: 4px;
}
.mcr-title {
    font-weight: 600;
    font-size: 14.5px;
    line-height: 1.3;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}
.mcr-artist { font-size: 13px; opacity: 0.85; }
.mcr-album  { font-size: 11.5px; opacity: 0.55; }
.mcr-meta-row {
    display: flex; flex-wrap: wrap; gap: 4px;
    margin-top: 2px;
}
.mcr-tag {
    font-size: 10.5px;
    padding: 1px 7px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.10);
    color: var(--body-text-color, inherit);
    opacity: 0.85;
}
.mcr-actions {
    display: flex; flex-wrap: wrap; gap: 8px;
    align-items: center;
    margin-top: 6px;
}
.mcr-actions audio {
    height: 32px;
    max-width: 220px;
}
.mcr-btn {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 12px; font-weight: 500;
    text-decoration: none;
    border: 1px solid currentColor;
    color: var(--body-text-color, inherit);
    transition: filter 0.15s ease, transform 0.15s ease;
}
.mcr-btn:hover { filter: brightness(1.05); transform: translateY(-1px); }
.mcr-btn.apple   { background: #fa233b; color: white; border-color: #fa233b; }
.mcr-btn.youtube { background: #ff0000; color: white; border-color: #ff0000; }
.mcr-yt details { margin-top: 6px; }
.mcr-yt summary {
    cursor: pointer;
    font-size: 12px;
    opacity: 0.7;
    list-style: none;
    user-select: none;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
}
.mcr-yt summary:hover { opacity: 1; background: rgba(128,128,128,0.10); }
.mcr-yt summary::-webkit-details-marker { display: none; }
.mcr-yt iframe {
    width: 100%;
    max-width: 480px;
    aspect-ratio: 16 / 9;
    height: auto;
    border: 0;
    border-radius: 10px;
    margin-top: 8px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.12);
}
.mcr-rank {
    flex: 0 0 24px;
    height: 24px; line-height: 24px;
    text-align: center;
    border-radius: 50%;
    font-size: 11px; font-weight: 600;
    background: linear-gradient(135deg, var(--mcr-grad-from), var(--mcr-grad-to));
    color: #ffffff;
    align-self: flex-start;
    box-shadow: 0 2px 6px rgba(67, 56, 202, 0.30);
}

/* Empty-state placeholder for the chat: shown by Gradio if implemented */
@media (max-width: 600px) {
    .mcr-art { flex: 0 0 72px; width: 72px; height: 72px; }
    .mcr-actions audio { max-width: 160px; }
}
</style>
"""


def _esc(s: str | None) -> str:
    return html.escape(s or "", quote=True)


def _render_card(rec: TrackRecommendation) -> str:
    t = rec.track
    art = (
        f'<img src="{_esc(t.album_art_url)}" alt="" loading="lazy"/>'
        if t.album_art_url else ""
    )

    actions: list[str] = []
    if t.preview_url:
        actions.append(
            f'<audio controls preload="none" src="{_esc(t.preview_url)}"></audio>'
        )
    if t.spotify_url:
        actions.append(
            f'<a class="mcr-btn apple" href="{_esc(t.spotify_url)}" '
            f'target="_blank" rel="noopener">Open in Apple Music</a>'
        )
    if rec.youtube and rec.youtube.watch_url:
        actions.append(
            f'<a class="mcr-btn youtube" href="{_esc(rec.youtube.watch_url)}" '
            f'target="_blank" rel="noopener">YouTube</a>'
        )

    yt_block = ""
    if rec.youtube and rec.youtube.embed_url:
        yt_block = (
            f'<div class="mcr-yt"><details>'
            f'<summary>▸ Play here</summary>'
            f'<iframe src="{_esc(rec.youtube.embed_url)}" '
            f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; '
            f'gyroscope; picture-in-picture" allowfullscreen></iframe>'
            f'</details></div>'
        )

    tags: list[str] = []
    if t.release_year:
        tags.append(f'<span class="mcr-tag">{_esc(str(t.release_year))}</span>')
    for g in (t.genres or [])[:2]:
        tags.append(f'<span class="mcr-tag">{_esc(g)}</span>')
    tag_row = (
        f'<div class="mcr-meta-row">{"".join(tags)}</div>' if tags else ""
    )

    return (
        f'<div class="mcr-card">'
        f'<div class="mcr-rank">{rec.rationale_rank}</div>'
        f'<div class="mcr-art">{art}</div>'
        f'<div class="mcr-body">'
        f'<div class="mcr-title">{_esc(t.name)}</div>'
        f'<div class="mcr-artist">{_esc(t.artist_str)}</div>'
        f'<div class="mcr-album">{_esc(t.album)}</div>'
        f'{tag_row}'
        f'<div class="mcr-actions">{"".join(actions)}</div>'
        f'{yt_block}'
        f'</div>'
        f'</div>'
    )


def render_response(resp: AgentResponse, show_meta: bool = True) -> str:
    """Render an AgentResponse as a single HTML string for ChatInterface."""
    wrap_id = f"mcr-{secrets.token_hex(4)}"
    parts: list[str] = [_CARD_CSS, f'<div id="{wrap_id}" class="mcr-wrap">']
    if resp.message:
        parts.append(f'<p class="mcr-message">{_esc(resp.message)}</p>')
    if resp.recommendations:
        parts.append('<div class="mcr-grid">')
        for rec in resp.recommendations:
            parts.append(_render_card(rec))
        parts.append('</div>')
    if show_meta and resp.recommendations:
        bits: list[str] = []
        if resp.timings_ms.get("total_ms") is not None:
            bits.append(f'<span class="mcr-meta-pill">⚡ {int(resp.timings_ms["total_ms"])} ms</span>')
        bits.append(f'<span class="mcr-meta-pill">🌈 H={resp.diversity_entropy:.2f}</span>')
        if resp.intent and resp.intent.diversify:
            bits.append('<span class="mcr-meta-pill">🔀 diversified</span>')
        if resp.intent and resp.intent.year_min:
            bits.append(
                f'<span class="mcr-meta-pill">🗓 {resp.intent.year_min}'
                f'{"–" + str(resp.intent.year_max) if resp.intent.year_max else "+"}</span>'
            )
        parts.append(f'<div class="mcr-meta">{"".join(bits)}</div>')
    parts.append("</div>")
    return "".join(parts)


def render_error(text: str) -> str:
    return (
        '<p style="color:#c0392b;background:rgba(192,57,43,0.08);'
        'padding:10px 14px;border-radius:8px;border:1px solid rgba(192,57,43,0.25);">'
        f'⚠️ {_esc(text)}</p>'
    )
