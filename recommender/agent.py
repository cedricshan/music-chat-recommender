"""Top-level orchestration: text in, (assistant_message, ranked tracks) out.

This is the single seam the UI talks to. Everything LLM- or API-specific
lives below it.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

from recommender.conversation import ConversationState, TurnRecord
from recommender.itunes_client import ItunesClient, build_itunes_query
from recommender.llm import GeminiLLM, LLMError, RerankResult, SearchQuery
from recommender.spotify_client import Track   # kept here for the Track dataclass only
from recommender.youtube_client import YouTubeClient, YouTubeHit


# --------------------------------------------------------------------------- #
#  Public response object.                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class TrackRecommendation:
    track: Track
    youtube: YouTubeHit | None
    rationale_rank: int  # 1-indexed position chosen by LLM


@dataclass
class AgentResponse:
    message: str
    recommendations: list[TrackRecommendation] = field(default_factory=list)
    intent: SearchQuery | None = None
    diversity_entropy: float = 0.0
    timings_ms: dict = field(default_factory=dict)
    debug_notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Helpers.                                                                   #
# --------------------------------------------------------------------------- #


def _shannon_entropy(items: list[str]) -> float:
    """Shannon entropy over a list of categorical labels (in nats)."""
    if not items:
        return 0.0
    counts = Counter(items)
    total = sum(counts.values())
    return -sum((c / total) * math.log(c / total) for c in counts.values() if c)


def _build_search_queries(intent: SearchQuery) -> list[tuple[str, str | None]]:
    """Turn one structured intent into 1-4 iTunes query plans.

    Returns a list of (query_text, attribute) pairs. `attribute` is iTunes'
    field-restriction param: `"songTerm"` and `"artistTerm"` narrow the
    search to just titles / artists respectively (None = search all fields).

    We fan out across facets so the candidate pool has variety:
      * mood + (top genre) — broad pull
      * one query per additional genre
      * one `attribute=artistTerm` query per seed artist
    Year filtering is applied after the search returns (iTunes has no
    server-side year filter).
    """
    plans: list[tuple[str, str | None]] = []
    moods = [m for m in intent.mood_keywords if m]
    genres = [g for g in intent.genres if g]
    seeds = [a for a in intent.seed_artists if a]

    # Always issue one broad combined query first, then narrower facets.
    if moods or genres:
        plans.append(
            (
                build_itunes_query(
                    keywords=moods[:2],
                    genres=genres[:1] if genres else None,
                ),
                None,
            )
        )
    # Diversify: prefer SHORT, single-token facet queries — iTunes' search
    # is keyword-overlap based, so concatenating many words tends to surface
    # stock/instrumental tracks instead of a varied real-artist pool.
    if intent.diversify:
        for g in genres[:3]:
            plans.append((g, None))
        for m in moods[:3]:
            plans.append((m, None))
    else:
        for g in genres[:2]:
            plans.append(
                (build_itunes_query(keywords=moods[:1] or None, genres=[g]), None)
            )
    for a in seeds[:2]:
        plans.append((a, "artistTerm"))
    if not plans:
        plans.append((build_itunes_query(keywords=["popular"]), None))

    seen: set[tuple[str, str | None]] = set()
    deduped: list[tuple[str, str | None]] = []
    for plan in plans:
        q = (plan[0].strip(), plan[1])
        if q[0] and q not in seen:
            seen.add(q)
            deduped.append(q)
    # Allow a few extra plans in diversify mode to give the rerank a richer pool.
    return deduped[: 6 if intent.diversify else 4]


def _filter_by_year(tracks: list[Track], lo: int | None, hi: int | None) -> list[Track]:
    """Drop tracks whose release_year is outside [lo, hi]; keep tracks with
    unknown year so we don't over-filter when iTunes omits the date."""
    if lo is None and hi is None:
        return tracks
    out: list[Track] = []
    for t in tracks:
        if t.release_year is None:
            out.append(t)
            continue
        if lo is not None and t.release_year < lo:
            continue
        if hi is not None and t.release_year > hi:
            continue
        out.append(t)
    return out


def _dedupe_tracks(track_lists: list[list[Track]]) -> list[Track]:
    seen: set[str] = set()
    out: list[Track] = []
    for tl in track_lists:
        for t in tl:
            if t.id in seen:
                continue
            seen.add(t.id)
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
#  Agent.                                                                     #
# --------------------------------------------------------------------------- #


class RecommenderAgent:
    """Single entrypoint: `respond(user_text, state)` -> AgentResponse."""

    def __init__(
        self,
        llm: GeminiLLM | None = None,
        music: ItunesClient | None = None,
        youtube: YouTubeClient | None = None,
    ) -> None:
        self.llm = llm or GeminiLLM()
        self.music = music or ItunesClient()
        self.youtube = youtube or YouTubeClient()

    def respond(self, user_text: str, state: ConversationState) -> AgentResponse:
        timings: dict = {}
        debug: list[str] = []
        t_total0 = time.perf_counter()

        # 1. Parse intent.
        try:
            t0 = time.perf_counter()
            intent = self.llm.parse_intent(
                user_text=user_text,
                history_summary=state.history_summary,
                must_exclude_artists=state.must_exclude_artists,
                liked_artists=state.liked_artists,
                recent_genres=state.recent_genres,
            )
            timings["parse_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        except LLMError as e:
            debug.append(f"parse_intent failed: {e}")
            return AgentResponse(
                message=(
                    "Sorry — I couldn't reach the language model just now. "
                    "Try again in a moment, or check that GEMINI_API_KEY is set."
                ),
                debug_notes=debug,
                timings_ms={"total_ms": round((time.perf_counter() - t_total0) * 1000, 1)},
            )

        # Always merge newly excluded artists into state immediately.
        if intent.new_exclude_artists:
            state.add_excluded_artists(intent.new_exclude_artists)

        # 2. Pure acknowledgement (no search) path.
        if intent.acknowledge_only and not (intent.mood_keywords or intent.genres or intent.seed_artists):
            msg_bits = []
            if intent.new_exclude_artists:
                msg_bits.append(
                    "Got it — I'll avoid "
                    + ", ".join(intent.new_exclude_artists)
                    + " from now on."
                )
            else:
                msg_bits.append("Noted. What would you like to hear next?")
            assistant_message = " ".join(msg_bits)
            state.record_turn(TurnRecord(user=user_text, assistant_summary=intent.assistant_intent_summary))
            timings["total_ms"] = round((time.perf_counter() - t_total0) * 1000, 1)
            return AgentResponse(
                message=assistant_message,
                intent=intent,
                timings_ms=timings,
                debug_notes=debug,
            )

        # 3. Search iTunes with one or more crafted queries.
        plans = _build_search_queries(intent)
        debug.append(f"plans={plans}")
        per_query_limit = max(15, min(50, intent.count * 6))
        track_lists: list[list[Track]] = []
        t0 = time.perf_counter()
        for q, attr in plans:
            try:
                track_lists.append(
                    self.music.search_tracks(q, limit=per_query_limit, attribute=attr)
                )
            except Exception as e:  # noqa: BLE001
                debug.append(f"search '{q}' failed: {e}")
        timings["search_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        candidates = _dedupe_tracks(track_lists)

        # 4. Apply year filter (iTunes has no server-side year filter).
        candidates = _filter_by_year(candidates, intent.year_min, intent.year_max)

        # 5. Filter excluded artists and tracks the user already saw.
        candidates = state.filter_excluded(candidates)
        if not intent.more_of_same:
            fresh = [t for t in candidates if t.id not in state.shown_track_ids]
            if len(fresh) >= intent.count:
                candidates = fresh

        if not candidates:
            timings["total_ms"] = round((time.perf_counter() - t_total0) * 1000, 1)
            state.record_turn(TurnRecord(user=user_text, assistant_summary=intent.assistant_intent_summary))
            return AgentResponse(
                message=(
                    "I searched a few angles but came up empty. "
                    "Try a different mood, genre, or remove some exclusions."
                ),
                intent=intent,
                timings_ms=timings,
                debug_notes=debug,
            )

        # 6. Cap pool. Smaller pool → faster rerank LLM call. iTunes already
        # gives us per-track genres at search time, so attach_genres is a
        # no-op (kept for symmetry with the legacy Spotify provider).
        pool = candidates[:20]
        self.music.attach_genres(pool)

        # 6. Ask the LLM to pick the final ordered subset + author the message.
        target_count = max(1, min(8, intent.count))
        try:
            t0 = time.perf_counter()
            rerank: RerankResult = self.llm.rerank(
                user_text=user_text,
                history_summary=state.history_summary,
                must_exclude_artists=state.must_exclude_artists,
                candidates=[t.to_dict() for t in pool],
                target_count=target_count,
                diversify=intent.diversify,
            )
            timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        except LLMError as e:
            debug.append(f"rerank failed, falling back to popularity: {e}")
            ordered = sorted(pool, key=lambda t: t.popularity, reverse=True)[:target_count]
            rerank = RerankResult(
                chosen_track_ids=[t.id for t in ordered],
                message="Here are some popular picks for that vibe.",
            )

        id_to_track = {t.id: t for t in pool}
        chosen: list[Track] = []
        for tid in rerank.chosen_track_ids:
            if tid in id_to_track and id_to_track[tid] not in chosen:
                chosen.append(id_to_track[tid])
        # Backfill if model under-delivered.
        if len(chosen) < target_count:
            for t in sorted(pool, key=lambda x: x.popularity, reverse=True):
                if t in chosen:
                    continue
                chosen.append(t)
                if len(chosen) >= target_count:
                    break
        chosen = chosen[:target_count]

        # 7. Resolve a YouTube embed for each chosen track (best-effort).
        recs: list[TrackRecommendation] = []
        t0 = time.perf_counter()
        for i, tr in enumerate(chosen, 1):
            yt: YouTubeHit | None = None
            try:
                yt = self.youtube.find_video(f"{tr.artist_str} - {tr.name} official audio")
            except Exception as e:  # noqa: BLE001
                debug.append(f"youtube lookup failed for {tr.id}: {e}")
            recs.append(TrackRecommendation(track=tr, youtube=yt, rationale_rank=i))
        timings["youtube_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # 8. Update state.
        all_genres = [g for tr in chosen for g in tr.genres]
        diversity = _shannon_entropy(all_genres)
        state.push_recent_genres(all_genres)
        state.record_turn(
            TurnRecord(
                user=user_text,
                assistant_summary=intent.assistant_intent_summary or rerank.message[:80],
                track_ids_shown=[tr.id for tr in chosen],
                artists_shown=[tr.artists[0] for tr in chosen if tr.artists],
                genres_shown=all_genres,
            )
        )

        timings["total_ms"] = round((time.perf_counter() - t_total0) * 1000, 1)
        return AgentResponse(
            message=rerank.message or "Here are a few that match.",
            recommendations=recs,
            intent=intent,
            diversity_entropy=diversity,
            timings_ms=timings,
            debug_notes=debug,
        )
