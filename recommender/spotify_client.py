"""Thin wrapper around the Spotify Web API.

Uses the Client Credentials flow (no user OAuth) which is the simplest way to
hit public read endpoints like /v1/search and /v1/tracks. The recommendations,
audio-features, audio-analysis and related-artists endpoints are unavailable
to apps registered after 2024-11-27, so this client deliberately exposes only
the endpoints that still work.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

import requests

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


@dataclass
class Track:
    """A normalized Spotify track view used by the rest of the system."""

    id: str
    name: str
    artists: list[str]
    album: str
    album_art_url: str | None
    preview_url: str | None
    spotify_url: str
    popularity: int
    duration_ms: int
    explicit: bool
    release_year: int | None
    genres: list[str] = field(default_factory=list)

    @property
    def artist_str(self) -> str:
        return ", ".join(self.artists)

    @property
    def search_query(self) -> str:
        """Best query string to find this track on YouTube etc."""
        return f"{self.artist_str} - {self.name}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "artists": self.artists,
            "album": self.album,
            "album_art_url": self.album_art_url,
            "preview_url": self.preview_url,
            "spotify_url": self.spotify_url,
            "popularity": self.popularity,
            "duration_ms": self.duration_ms,
            "explicit": self.explicit,
            "release_year": self.release_year,
            "genres": self.genres,
        }


class SpotifyError(RuntimeError):
    pass


class SpotifyClient:
    """Caches the app token in memory and refreshes it before expiry."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET", "")
        if not self._client_id or not self._client_secret:
            raise SpotifyError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set "
                "(in the environment, .env, or passed explicitly)."
            )
        self._timeout = timeout
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._lock = threading.Lock()
        self._artist_genre_cache: dict[str, list[str]] = {}

    def _fetch_token(self) -> str:
        creds = f"{self._client_id}:{self._client_secret}".encode("utf-8")
        b64 = base64.b64encode(creds).decode("ascii")
        resp = requests.post(
            SPOTIFY_TOKEN_URL,
            headers={
                "Authorization": f"Basic {b64}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            raise SpotifyError(
                f"Spotify token request failed ({resp.status_code}): {resp.text[:300]}"
            )
        body = resp.json()
        token = body["access_token"]
        expires_in = int(body.get("expires_in", 3600))
        self._token = token
        # Refresh 60s before actual expiry to avoid races.
        self._token_expires_at = time.time() + expires_in - 60
        return token

    def _token_value(self) -> str:
        with self._lock:
            if self._token is None or time.time() >= self._token_expires_at:
                return self._fetch_token()
            return self._token

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{SPOTIFY_API_BASE}{path}"
        for attempt in range(2):
            token = self._token_value()
            resp = requests.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=self._timeout,
            )
            if resp.status_code == 401 and attempt == 0:
                # Token may be stale; force refresh once.
                with self._lock:
                    self._token = None
                continue
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "1"))
                time.sleep(min(retry_after, 5))
                continue
            if resp.status_code >= 400:
                raise SpotifyError(
                    f"Spotify GET {path} failed ({resp.status_code}): {resp.text[:300]}"
                )
            return resp.json()
        raise SpotifyError(f"Spotify GET {path} failed after retries.")

    @staticmethod
    def _parse_track(item: dict) -> Track:
        album = item.get("album", {}) or {}
        images = album.get("images") or []
        album_art = images[0]["url"] if images else None
        release_date = album.get("release_date") or ""
        release_year: int | None
        try:
            release_year = int(release_date[:4]) if release_date else None
        except ValueError:
            release_year = None
        return Track(
            id=item["id"],
            name=item["name"],
            artists=[a["name"] for a in item.get("artists", [])],
            album=album.get("name", ""),
            album_art_url=album_art,
            preview_url=item.get("preview_url"),
            spotify_url=(item.get("external_urls") or {}).get(
                "spotify", f"https://open.spotify.com/track/{item['id']}"
            ),
            popularity=int(item.get("popularity", 0)),
            duration_ms=int(item.get("duration_ms", 0)),
            explicit=bool(item.get("explicit", False)),
            release_year=release_year,
        )

    def search_tracks(
        self,
        query: str,
        limit: int = 20,
        market: str | None = "US",
    ) -> list[Track]:
        """Search for tracks. `query` may include Spotify advanced filters
        such as `genre:"indie pop"`, `year:2010-2020`, `artist:"radiohead"`.
        """
        params: dict = {"q": query, "type": "track", "limit": min(max(limit, 1), 50)}
        if market:
            params["market"] = market
        body = self._get("/search", params=params)
        items = ((body.get("tracks") or {}).get("items")) or []
        return [self._parse_track(i) for i in items if i and i.get("id")]

    def get_artist_genres(self, artist_ids: Iterable[str]) -> dict[str, list[str]]:
        """Fetch genres for a batch of artists, cached per process."""
        unique_ids = [a for a in dict.fromkeys(artist_ids) if a]
        missing = [a for a in unique_ids if a not in self._artist_genre_cache]
        for i in range(0, len(missing), 50):
            chunk = missing[i : i + 50]
            body = self._get("/artists", params={"ids": ",".join(chunk)})
            for art in body.get("artists") or []:
                if not art:
                    continue
                self._artist_genre_cache[art["id"]] = list(art.get("genres") or [])
        return {a: self._artist_genre_cache.get(a, []) for a in unique_ids}

    def attach_genres(self, tracks: list[Track]) -> list[Track]:
        """Mutate-and-return: enrich each Track with genres pulled from the
        first artist that has any. Done in batches to stay under the rate limit.
        """
        if not tracks:
            return tracks
        # We need raw artist ids; the parsed Track only stores names. Re-fetch
        # the tracks endpoint in batch to get the artist id list cheaply.
        ids = [t.id for t in tracks]
        body = self._get("/tracks", params={"ids": ",".join(ids[:50])})
        raw_tracks = body.get("tracks") or []
        artist_ids_per_track: list[list[str]] = []
        for raw in raw_tracks:
            if not raw:
                artist_ids_per_track.append([])
                continue
            artist_ids_per_track.append([a["id"] for a in raw.get("artists", []) if a])
        all_artist_ids = [aid for sub in artist_ids_per_track for aid in sub]
        genre_map = self.get_artist_genres(all_artist_ids)
        for track, artist_ids in zip(tracks, artist_ids_per_track):
            collected: list[str] = []
            for aid in artist_ids:
                for g in genre_map.get(aid, []):
                    if g not in collected:
                        collected.append(g)
            track.genres = collected
        return tracks


def build_query(
    keywords: Iterable[str] | None = None,
    genres: Iterable[str] | None = None,
    artists_include: Iterable[str] | None = None,
    year_range: tuple[int | None, int | None] | None = None,
) -> str:
    """Compose a Spotify search query from structured pieces.

    Spotify treats space-separated tokens as AND. Quoted artist/genre values
    keep multi-word names intact. Kept here (not in Agent) so it can be unit
    tested independently of the LLM.
    """
    parts: list[str] = []
    for kw in keywords or []:
        kw = kw.strip()
        if kw:
            parts.append(kw)
    for g in genres or []:
        g = g.strip()
        if g:
            parts.append(f'genre:"{g}"')
    for a in artists_include or []:
        a = a.strip()
        if a:
            parts.append(f'artist:"{a}"')
    if year_range and (year_range[0] or year_range[1]):
        lo = year_range[0] or 1900
        hi = year_range[1] or 2100
        parts.append(f"year:{lo}-{hi}")
    return " ".join(parts).strip()


if __name__ == "__main__":
    # Manual smoke test: load .env, run a sad-songs query, print first 3.
    from dotenv import load_dotenv

    load_dotenv()
    client = SpotifyClient()
    q = build_query(keywords=["melancholy", "rain"], genres=["indie"], year_range=(2015, 2024))
    print(f"Query: {q}")
    results = client.search_tracks(q, limit=5)
    client.attach_genres(results)
    for t in results:
        print(f"- {t.artist_str} — {t.name}  ({t.album}, {t.release_year}) "
              f"pop={t.popularity}  preview={'Y' if t.preview_url else 'N'}  "
              f"genres={t.genres[:3]}")
