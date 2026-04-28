"""Thin wrapper around Apple's iTunes Search API.

Why iTunes and not Spotify: as of 2026-03 Spotify's Web API requires the
*owner* of the Developer App to hold an active Spotify Premium subscription,
so its Client Credentials flow returns 403 for free accounts. The iTunes
Search API is fully public, requires no authentication, no API key, no
OAuth, and returns 30s previews + cover art + Apple Music share links —
exactly the fields the rest of the system already expects.

API shape we use:
    GET https://itunes.apple.com/search?term=...&entity=song&limit=...
    Optional `attribute=songTerm|artistTerm` narrows the search field.

Docs: https://performance-partners.apple.com/search-api
Limits: ~20 calls/min per IP (no documented daily cap). We cache per query.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Iterable

import requests

from recommender.spotify_client import Track, SpotifyError as APIError

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"

# Compilation/cover/karaoke clutter we want to suppress.
_NOISE_PATTERNS = [
    re.compile(r"\(instrumental\)", re.I),
    re.compile(r"\binstrumental version\b", re.I),
    re.compile(r"\bkaraoke\b", re.I),
    re.compile(r"\bcover( version)?\b", re.I),
    re.compile(r"\b(originally performed|in the style of|tribute to)\b", re.I),
    re.compile(r"\b(string quartet|orchestral|piano version)\b", re.I),
]
_NOISE_ARTISTS = {
    "various artists",
    "guitar dreamers",
    "acoustic heartstrings",
    "vitamin string quartet",
    "midnite string quartet",
    "the karaoke channel",
    "the hit crew",
    "the sing-along music makers",
}


def _looks_like_noise(track_name: str, artist_name: str, album_name: str) -> bool:
    if (artist_name or "").strip().lower() in _NOISE_ARTISTS:
        return True
    blob = f"{track_name} | {album_name}"
    return any(p.search(blob) for p in _NOISE_PATTERNS)


def _bigger_artwork(url: str | None, target: int = 400) -> str | None:
    """iTunes returns 100x100 by default; bump to 300/400/600 by URL hack."""
    if not url:
        return url
    return re.sub(r"\d+x\d+bb\.(jpg|png)", f"{target}x{target}bb.\\1", url)


class ItunesClient:
    """Stateless except for an LRU and a soft rate limiter."""

    def __init__(self, country: str = "US", timeout: float = 10.0) -> None:
        self._country = country
        self._timeout = timeout
        self._cache: dict[tuple, list[Track]] = {}
        self._lock = threading.Lock()
        self._last_request_at: float = 0.0
        self._min_interval_s: float = 0.4   # ~150 req/min, well under the soft limit

    def _throttle(self) -> None:
        with self._lock:
            now = time.time()
            wait = self._min_interval_s - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.time()

    @staticmethod
    def _parse(item: dict) -> Track | None:
        if (item.get("kind") or item.get("wrapperType")) not in (
            "song",
            "track",
        ) and item.get("kind") != "song":
            # Skip podcasts / audiobooks / etc.
            return None
        track_name = item.get("trackName") or ""
        artist_name = item.get("artistName") or ""
        album_name = item.get("collectionName") or ""
        if not track_name or not artist_name:
            return None
        if _looks_like_noise(track_name, artist_name, album_name):
            return None
        track_id = str(item.get("trackId") or "")
        if not track_id:
            return None
        release_date = item.get("releaseDate") or ""
        try:
            release_year = int(release_date[:4]) if release_date else None
        except ValueError:
            release_year = None
        primary_genre = item.get("primaryGenreName") or ""
        return Track(
            id=track_id,
            name=track_name,
            artists=[artist_name],
            album=album_name,
            album_art_url=_bigger_artwork(item.get("artworkUrl100")),
            preview_url=item.get("previewUrl"),
            spotify_url=item.get("trackViewUrl") or "",
            popularity=0,           # iTunes doesn't expose a popularity score
            duration_ms=int(item.get("trackTimeMillis", 0) or 0),
            explicit=(item.get("trackExplicitness") == "explicit"),
            release_year=release_year,
            genres=[primary_genre] if primary_genre else [],
        )

    def search_tracks(
        self,
        query: str,
        limit: int = 20,
        attribute: str | None = None,
    ) -> list[Track]:
        if not query.strip():
            return []
        cache_key = (query.strip().lower(), limit, attribute or "", self._country)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        params = {
            "term": query,
            "media": "music",
            "entity": "song",
            "limit": min(max(limit, 1), 100),
            "country": self._country,
        }
        if attribute:
            params["attribute"] = attribute
        self._throttle()
        try:
            resp = requests.get(ITUNES_SEARCH_URL, params=params, timeout=self._timeout)
        except requests.RequestException as e:
            raise APIError(f"iTunes search '{query}' failed: {e}") from e
        if resp.status_code == 403:
            raise APIError(
                f"iTunes search rejected ({resp.status_code}). "
                "Likely rate-limited; back off and retry."
            )
        if resp.status_code >= 400:
            raise APIError(
                f"iTunes search '{query}' failed ({resp.status_code}): {resp.text[:200]}"
            )
        body = resp.json()
        tracks: list[Track] = []
        for item in body.get("results", []) or []:
            t = self._parse(item)
            if t is not None:
                tracks.append(t)
        # Dedupe by (artist, name) — iTunes often returns the same song on
        # multiple albums (single, album, deluxe edition).
        seen: set[tuple[str, str]] = set()
        deduped: list[Track] = []
        for t in tracks:
            key = (t.artists[0].lower(), t.name.lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(t)
        with self._lock:
            self._cache[cache_key] = deduped
        return deduped

    def attach_genres(self, tracks: list[Track]) -> list[Track]:
        """No-op for iTunes — genre is parsed at search time. Kept so the
        agent can call this method uniformly across providers."""
        return tracks


def build_itunes_query(
    keywords: Iterable[str] | None = None,
    genres: Iterable[str] | None = None,
    artists_include: Iterable[str] | None = None,
) -> str:
    """Compose a free-text iTunes query.

    iTunes does NOT support `genre:`/`year:`/`artist:` syntax — its search
    is just space-separated keywords scored by tf-idf-ish relevance. So we
    just append all the tokens. Year filtering happens client-side after the
    search returns (see Agent post-filtering).
    """
    parts: list[str] = []
    for kw in keywords or []:
        kw = (kw or "").strip()
        if kw:
            parts.append(kw)
    for g in genres or []:
        g = (g or "").strip()
        if g:
            parts.append(g)
    for a in artists_include or []:
        a = (a or "").strip()
        if a:
            parts.append(a)
    return " ".join(parts).strip()


if __name__ == "__main__":
    c = ItunesClient()
    print("== free text 'rainy night indie':")
    for t in c.search_tracks("rainy night indie", limit=5):
        print(f"  - {t.artist_str} -- {t.name} ({t.release_year}) genre={t.genres}")
    print("== artist search 'Bon Iver':")
    for t in c.search_tracks("Bon Iver", limit=5, attribute="artistTerm"):
        print(f"  - {t.artist_str} -- {t.name} ({t.release_year}) genre={t.genres}")
