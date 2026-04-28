"""Unit tests for the pure logic that does NOT require network/API keys."""

from __future__ import annotations

import math

from pydantic import ValidationError
import pytest

from recommender.agent import (
    _build_search_queries,
    _dedupe_tracks,
    _filter_by_year,
    _shannon_entropy,
)
from recommender.conversation import ConversationState, TurnRecord
from recommender.itunes_client import build_itunes_query
from recommender.llm import RerankResult, SearchQuery
from recommender.spotify_client import Track


# ----------------------- ConversationState ------------------------------- #


def test_excluded_artists_dedupe_case_insensitive():
    s = ConversationState()
    s.add_excluded_artists(["Coldplay", "coldplay", "  COLDPLAY  ", "Taylor Swift"])
    assert s.must_exclude_artists == ["Coldplay", "Taylor Swift"]


def test_filter_excluded_skips_any_artist():
    s = ConversationState()
    s.add_excluded_artists(["Drake"])
    tracks = [
        _track("a", ["Future", "Drake"]),
        _track("b", ["Frank Ocean"]),
        _track("c", ["Drake"]),
    ]
    kept = s.filter_excluded(tracks)
    assert [t.id for t in kept] == ["b"]


def test_recent_genres_window_lowercase_dedupe():
    s = ConversationState()
    s.push_recent_genres(["Indie Pop", "indie pop", "AMBIENT"])
    assert s.recent_genres == ["indie pop", "ambient"]


def test_record_turn_summary_grows():
    s = ConversationState()
    s.record_turn(TurnRecord(user="a", assistant_summary="A"))
    s.record_turn(TurnRecord(user="b", assistant_summary="B"))
    assert "turn1" in s.history_summary and "turn2" in s.history_summary


# ------------------------------ Schemas --------------------------------- #


def test_search_query_defaults():
    q = SearchQuery()
    assert q.count == 5 and q.diversify is False and q.acknowledge_only is False


def test_search_query_validates_types():
    with pytest.raises(ValidationError):
        SearchQuery(count="five")  # type: ignore[arg-type]


def test_rerank_result_round_trip():
    r = RerankResult(chosen_track_ids=["a", "b"], message="hi")
    assert RerankResult.model_validate(r.model_dump()) == r


# -------------------------- query construction --------------------------- #


def test_build_itunes_query_concatenates():
    q = build_itunes_query(
        keywords=["rainy night"],
        genres=["indie pop"],
        artists_include=["Phoebe Bridgers"],
    )
    assert "rainy night" in q
    assert "indie pop" in q
    assert "Phoebe Bridgers" in q


def test_build_search_queries_fans_out():
    intent = SearchQuery(
        mood_keywords=["melancholy"],
        genres=["indie", "ambient"],
        seed_artists=["Bon Iver"],
        year_min=2015,
        year_max=2024,
    )
    plans = _build_search_queries(intent)
    # Each plan is (query, attribute_or_None).
    queries = [q for q, _ in plans]
    assert any("indie" in q for q in queries)
    # Seed-artist plan must use the artistTerm attribute and contain the name.
    assert any(q == "Bon Iver" and a == "artistTerm" for q, a in plans)


def test_build_search_queries_falls_back_when_empty():
    plans = _build_search_queries(SearchQuery())
    assert len(plans) == 1
    q, attr = plans[0]
    assert q and attr is None


def test_filter_by_year_keeps_unknown_years():
    tracks = [
        _track("a"),                      # year=2020
        _make(_track("b"), release_year=2010),
        _make(_track("c"), release_year=None),
    ]
    out = _filter_by_year(tracks, lo=2015, hi=2024)
    assert {t.id for t in out} == {"a", "c"}


def _make(t: Track, **changes) -> Track:
    """Return a copy of t with the named attributes overridden."""
    from dataclasses import replace
    return replace(t, **changes)


# ---------------------------- helpers ------------------------------------ #


def test_dedupe_tracks_preserves_order():
    a, b, c = _track("a"), _track("b"), _track("c")
    out = _dedupe_tracks([[a, b], [b, c], [a]])
    assert [t.id for t in out] == ["a", "b", "c"]


def test_shannon_entropy_matches_definition():
    # uniform over 4 distinct items → ln(4)
    val = _shannon_entropy(["a", "b", "c", "d"])
    assert abs(val - math.log(4)) < 1e-9
    # constant list → 0
    assert _shannon_entropy(["x", "x", "x"]) == 0.0
    # empty → 0
    assert _shannon_entropy([]) == 0.0


# ------------------------------ fixtures --------------------------------- #


def _track(tid: str, artists: list[str] | None = None) -> Track:
    return Track(
        id=tid,
        name=f"name-{tid}",
        artists=artists or ["Generic Artist"],
        album="Album",
        album_art_url=None,
        preview_url=None,
        spotify_url=f"https://open.spotify.com/track/{tid}",
        popularity=50,
        duration_ms=200_000,
        explicit=False,
        release_year=2020,
        genres=["indie"],
    )
