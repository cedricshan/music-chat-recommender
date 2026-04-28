"""Look up an embeddable YouTube videoId for a given track.

Uses YouTube Data API v3 `search.list`. A single search costs 100 quota units;
the daily free quota is 10,000 units (~100 lookups). We cache aggressively in
memory and fall back to a simple `youtube.com/results?search_query=...` link
when the API is missing or quota is exhausted, so the chatbot still works.
"""

from __future__ import annotations

import os
import threading
import urllib.parse
from dataclasses import dataclass

import requests

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


@dataclass
class YouTubeHit:
    video_id: str | None
    title: str | None
    watch_url: str
    embed_url: str | None

    @property
    def has_embed(self) -> bool:
        return self.embed_url is not None


def _fallback_search_url(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)


class YouTubeClient:
    """Thin wrapper. If no API key is configured we still return a usable
    'open YouTube search' link so the chatbot degrades gracefully."""

    def __init__(self, api_key: str | None = None, timeout: float = 10.0) -> None:
        self._api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")
        self._timeout = timeout
        self._cache: dict[str, YouTubeHit] = {}
        self._lock = threading.Lock()
        self._quota_exhausted = False

    @property
    def enabled(self) -> bool:
        return bool(self._api_key) and not self._quota_exhausted

    def find_video(self, query: str) -> YouTubeHit:
        """Return a YouTubeHit for the given free-text query.

        Always returns; if the API isn't usable the hit will only contain a
        watch_url that opens YouTube search results in a new tab.
        """
        key = query.strip().lower()
        if not key:
            return YouTubeHit(video_id=None, title=None, watch_url=_fallback_search_url(query), embed_url=None)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        if not self.enabled:
            hit = YouTubeHit(
                video_id=None,
                title=None,
                watch_url=_fallback_search_url(query),
                embed_url=None,
            )
            with self._lock:
                self._cache[key] = hit
            return hit
        try:
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": 1,
                "videoEmbeddable": "true",
                "safeSearch": "none",
                "key": self._api_key,
            }
            resp = requests.get(
                f"{YOUTUBE_API_BASE}/search", params=params, timeout=self._timeout
            )
            if resp.status_code == 403:
                # Most often quota exceeded; flip the breaker so subsequent
                # calls in this process skip straight to the fallback.
                self._quota_exhausted = True
                hit = YouTubeHit(
                    video_id=None,
                    title=None,
                    watch_url=_fallback_search_url(query),
                    embed_url=None,
                )
                with self._lock:
                    self._cache[key] = hit
                return hit
            resp.raise_for_status()
            items = resp.json().get("items") or []
            if not items:
                hit = YouTubeHit(
                    video_id=None,
                    title=None,
                    watch_url=_fallback_search_url(query),
                    embed_url=None,
                )
            else:
                first = items[0]
                vid = first.get("id", {}).get("videoId")
                title = first.get("snippet", {}).get("title")
                if vid:
                    hit = YouTubeHit(
                        video_id=vid,
                        title=title,
                        watch_url=f"https://www.youtube.com/watch?v={vid}",
                        embed_url=f"https://www.youtube.com/embed/{vid}",
                    )
                else:
                    hit = YouTubeHit(
                        video_id=None,
                        title=title,
                        watch_url=_fallback_search_url(query),
                        embed_url=None,
                    )
        except requests.RequestException:
            hit = YouTubeHit(
                video_id=None,
                title=None,
                watch_url=_fallback_search_url(query),
                embed_url=None,
            )
        with self._lock:
            self._cache[key] = hit
        return hit


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    client = YouTubeClient()
    hit = client.find_video("Radiohead - No Surprises official")
    print(hit)
