"""Per-session conversation memory.

We do NOT persist this across sessions on purpose: the deployed Hugging Face
Space has no database, and ephemeral state keeps the privacy story simple.
The Gradio app holds one ConversationState per browser tab via `gr.State`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


def _normalize_artist(name: str) -> str:
    """Case-insensitive, trimmed artist key for set membership."""
    return name.strip().lower()


@dataclass
class TurnRecord:
    user: str
    assistant_summary: str
    track_ids_shown: list[str] = field(default_factory=list)
    artists_shown: list[str] = field(default_factory=list)
    genres_shown: list[str] = field(default_factory=list)


@dataclass
class ConversationState:
    """Mutable bag of per-session memory consumed by the agent."""

    must_exclude_artists: list[str] = field(default_factory=list)
    liked_artists: list[str] = field(default_factory=list)
    recent_genres: list[str] = field(default_factory=list)  # last ~10
    history_summary: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    shown_track_ids: set[str] = field(default_factory=set)

    # ---------------------------- mutators -------------------------------- #

    def add_excluded_artists(self, names: Iterable[str]) -> None:
        existing = {_normalize_artist(a) for a in self.must_exclude_artists}
        for n in names:
            if not n:
                continue
            key = _normalize_artist(n)
            if key and key not in existing:
                self.must_exclude_artists.append(n.strip())
                existing.add(key)

    def add_liked_artists(self, names: Iterable[str]) -> None:
        existing = {_normalize_artist(a) for a in self.liked_artists}
        for n in names:
            if not n:
                continue
            key = _normalize_artist(n)
            if key and key not in existing:
                self.liked_artists.append(n.strip())
                existing.add(key)

    def push_recent_genres(self, genres: Iterable[str]) -> None:
        for g in genres:
            g = g.strip().lower()
            if not g:
                continue
            if g in self.recent_genres:
                self.recent_genres.remove(g)
            self.recent_genres.append(g)
        # Keep window bounded.
        if len(self.recent_genres) > 12:
            self.recent_genres = self.recent_genres[-12:]

    def record_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)
        for tid in turn.track_ids_shown:
            self.shown_track_ids.add(tid)
        if len(self.turns) > 30:
            # Keep memory bounded across long sessions.
            self.turns = self.turns[-30:]
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        """Cheap deterministic summary; the LLM also sees the last 3 raw turns."""
        recent = self.turns[-3:]
        bits: list[str] = []
        for i, t in enumerate(recent, 1):
            short_user = t.user.strip().replace("\n", " ")
            if len(short_user) > 80:
                short_user = short_user[:77] + "..."
            bits.append(f"turn{i}: user='{short_user}' summary='{t.assistant_summary}'")
        self.history_summary = " | ".join(bits)

    # --------------------------- read helpers ----------------------------- #

    def is_artist_excluded(self, name: str) -> bool:
        key = _normalize_artist(name)
        return any(_normalize_artist(a) == key for a in self.must_exclude_artists)

    def filter_excluded(self, tracks: list) -> list:
        """Drop any track whose primary artist is excluded."""
        if not self.must_exclude_artists:
            return tracks
        out = []
        excluded = {_normalize_artist(a) for a in self.must_exclude_artists}
        for t in tracks:
            primary = (t.artists[0] if getattr(t, "artists", None) else "") or ""
            if _normalize_artist(primary) in excluded:
                continue
            # Also drop if ANY listed artist is excluded.
            if any(_normalize_artist(a) in excluded for a in t.artists):
                continue
            out.append(t)
        return out

    def to_compact_dict(self) -> dict:
        return {
            "must_exclude_artists": list(self.must_exclude_artists),
            "liked_artists": list(self.liked_artists),
            "recent_genres": list(self.recent_genres),
            "history_summary": self.history_summary,
            "n_turns": len(self.turns),
            "n_shown_tracks": len(self.shown_track_ids),
        }
