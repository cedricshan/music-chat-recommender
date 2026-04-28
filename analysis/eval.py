"""Quantitative evaluation of the recommender, used by the QMD report.

Three experiments:

1. **Intent-parser accuracy**: run the LLM intent parser on the labeled
   set in `analysis.labeled_intents` and compute per-criterion accuracy
   plus an aggregate score.

2. **Diversity ablation**: for a fixed set of representative prompts,
   compare the genre-level Shannon entropy of (a) popularity-only ranking
   of the candidate pool vs (b) the LLM rerank that includes the
   `diversify=true` instruction. We expect the LLM rerank to win when
   diversify is requested.

3. **Latency distribution**: end-to-end response time over all prompts
   from experiment 2, broken down by stage.

Outputs are written to `analysis/eval_results.json` and the figures to
`analysis/figures/`. Run this once after collecting data; the QMD
consumes the cached files so the report renders deterministically.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from analysis.labeled_intents import LABELED
from recommender.agent import RecommenderAgent, _shannon_entropy
from recommender.conversation import ConversationState
from recommender.itunes_client import ItunesClient
from recommender.llm import GeminiLLM
from recommender.youtube_client import YouTubeClient

ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_PATH = ANALYSIS_DIR / "eval_results.json"
FIGURES_DIR = ANALYSIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ----------------------------- Experiment 1 ------------------------------ #


@dataclass
class IntentEvalRow:
    utterance: str
    expected: dict
    predicted: dict
    passed_criteria: dict
    overall_pass: bool
    parse_ms: float


def _check_criteria(predicted: dict, expected: dict) -> dict[str, bool]:
    """Apply a small DSL over the expected dict.

    Supported keys (all optional):
        mood_keywords_nonempty: bool
        genres_contains_any: list[str]
        seed_artists_contains_any: list[str]
        new_exclude_artists_contains_any: list[str]
        year_min_lte: int        (predicted.year_min must be <= this)
        year_max_gte: int        (predicted.year_max must be >= this)
        count: int               (exact equality)
        diversify: bool
        more_of_same: bool
        acknowledge_only: bool
    """
    results: dict[str, bool] = {}
    p = predicted

    def _lower_set(xs):
        return {str(x).lower().strip() for x in (xs or [])}

    for k, v in expected.items():
        try:
            if k == "mood_keywords_nonempty":
                results[k] = bool(p.get("mood_keywords")) == bool(v)
            elif k == "genres_contains_any":
                results[k] = bool(_lower_set(p.get("genres")) & _lower_set(v))
            elif k == "seed_artists_contains_any":
                results[k] = bool(_lower_set(p.get("seed_artists")) & _lower_set(v))
            elif k == "new_exclude_artists_contains_any":
                results[k] = bool(_lower_set(p.get("new_exclude_artists")) & _lower_set(v))
            elif k == "year_min_lte":
                ymin = p.get("year_min")
                results[k] = ymin is not None and ymin <= v
            elif k == "year_max_gte":
                ymax = p.get("year_max")
                results[k] = ymax is not None and ymax >= v
            elif k == "count":
                results[k] = int(p.get("count", -1)) == v
            elif k in {"diversify", "more_of_same", "acknowledge_only"}:
                results[k] = bool(p.get(k)) == bool(v)
            else:
                results[k] = False
        except Exception:
            results[k] = False
    return results


def run_intent_eval(llm: GeminiLLM, throttle_s: float = 0.5) -> list[IntentEvalRow]:
    """Light throttle: Groq's free tier is 30 RPM (one call every 2s is the
    floor); 0.5s gives us a comfortable margin while keeping the run fast."""
    rows: list[IntentEvalRow] = []
    for i, (utt, exp) in enumerate(LABELED):
        if i > 0:
            time.sleep(throttle_s)
        t0 = time.perf_counter()
        try:
            parsed = llm.parse_intent(
                user_text=utt,
                history_summary="",
                must_exclude_artists=[],
                liked_artists=[],
                recent_genres=[],
            )
            pred = parsed.model_dump()
            err = None
        except Exception as e:  # noqa: BLE001
            pred = {}
            err = str(e)
        elapsed = (time.perf_counter() - t0) * 1000
        crit = _check_criteria(pred, exp) if not err else {k: False for k in exp}
        rows.append(
            IntentEvalRow(
                utterance=utt,
                expected=exp,
                predicted=pred if not err else {"_error": err},
                passed_criteria=crit,
                overall_pass=all(crit.values()) if crit else False,
                parse_ms=elapsed,
            )
        )
        if err:
            print(f"   [{i+1}/{len(LABELED)}] WARN error on {utt!r}: {err[:80]}", flush=True)
        else:
            mark = "OK" if all(crit.values()) else "PARTIAL"
            print(f"   [{i+1}/{len(LABELED)}] {mark} {utt[:60]!r}", flush=True)
    return rows


# ----------------------------- Experiment 2 ------------------------------ #


DIVERSITY_PROMPTS: list[tuple[str, bool]] = [
    ("Recommend 6 sad nighttime songs.", False),
    ("Recommend 6 sad nighttime songs, mix it up across genres.", True),
    ("6 upbeat indie tracks from 2018-2022.", False),
    ("6 upbeat indie tracks from 2018-2022, very diverse.", True),
    ("6 chill electronic tracks for studying.", False),
    ("6 chill electronic tracks for studying, surprise me with variety.", True),
]


@dataclass
class DiversityRow:
    prompt: str
    requested_diverse: bool
    n_candidates: int
    n_returned: int
    pop_rank_entropy: float
    llm_rank_entropy: float
    total_ms: float
    parse_ms: float
    search_ms: float
    rerank_ms: float
    youtube_ms: float


def run_diversity_eval(agent: RecommenderAgent, throttle_s: float = 1.0) -> list[DiversityRow]:
    """Each prompt burns 2 LLM calls (parse + rerank). Even so, Groq's
    1000 RPD on free tier easily covers the 12 calls this loop generates."""
    rows: list[DiversityRow] = []
    for i, (prompt, want_diverse) in enumerate(DIVERSITY_PROMPTS):
        if i > 0:
            time.sleep(throttle_s)
        # Each prompt gets its own conversation so they don't interfere.
        state = ConversationState()
        try:
            resp = agent.respond(prompt, state)
        except Exception as e:  # noqa: BLE001
            print(f"   [{i+1}/{len(DIVERSITY_PROMPTS)}] WARN prompt failed: {prompt[:50]!r} -> {e}", flush=True)
            continue
        if not resp.recommendations:
            print(f"   [{i+1}/{len(DIVERSITY_PROMPTS)}] WARN no recs for {prompt[:50]!r}", flush=True)
            continue
        print(f"   [{i+1}/{len(DIVERSITY_PROMPTS)}] OK {prompt[:50]!r} -> "
              f"{len(resp.recommendations)} tracks, H_llm={resp.diversity_entropy:.2f}", flush=True)

        # Recompute popularity-rank baseline from the candidate pool we'd have
        # had: union of recommended tracks isn't the pool, so we re-search to
        # rebuild a comparable pool. To keep this cheap, we approximate by
        # running the same intent-derived queries inline.
        intent = resp.intent
        if intent is None:
            continue
        from recommender.agent import _build_search_queries, _dedupe_tracks, _filter_by_year

        plans = _build_search_queries(intent)
        pool_lists = []
        for q, attr in plans:
            try:
                pool_lists.append(
                    agent.music.search_tracks(q, limit=30, attribute=attr)
                )
            except Exception:
                pass
        pool = _filter_by_year(_dedupe_tracks(pool_lists), intent.year_min, intent.year_max)
        pool = state.filter_excluded(pool)[:30]
        agent.music.attach_genres(pool)
        # iTunes does not return a popularity score, so we use a stable proxy
        # (newer release date wins) for the popularity-rank baseline, which
        # mirrors what a naive "most-recent" recommender would do.
        pop_sort_key = lambda t: (t.release_year or 0)

        target = len(resp.recommendations)
        pop_rank = sorted(pool, key=pop_sort_key, reverse=True)[:target]
        pop_genres = [g for t in pop_rank for g in t.genres]
        llm_genres = [g for r in resp.recommendations for g in r.track.genres]

        rows.append(
            DiversityRow(
                prompt=prompt,
                requested_diverse=want_diverse,
                n_candidates=len(pool),
                n_returned=target,
                pop_rank_entropy=_shannon_entropy(pop_genres),
                llm_rank_entropy=_shannon_entropy(llm_genres),
                total_ms=resp.timings_ms.get("total_ms", 0.0),
                parse_ms=resp.timings_ms.get("parse_ms", 0.0),
                search_ms=resp.timings_ms.get("search_ms", 0.0),
                rerank_ms=resp.timings_ms.get("rerank_ms", 0.0),
                youtube_ms=resp.timings_ms.get("youtube_ms", 0.0),
            )
        )
    return rows


# ------------------------------- Plotting -------------------------------- #


def make_plots(intent_rows: list[IntentEvalRow], diversity_rows: list[DiversityRow]) -> dict:
    """Return dict mapping figure name to relative file path (saved as PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, str] = {}

    # 1. Intent-parser accuracy by criterion key.
    crit_totals: dict[str, list[bool]] = {}
    for r in intent_rows:
        for k, v in r.passed_criteria.items():
            crit_totals.setdefault(k, []).append(v)
    if crit_totals:
        labels = sorted(crit_totals.keys())
        values = [sum(crit_totals[k]) / len(crit_totals[k]) for k in labels]
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(labels) + 1.5))
        ax.barh(labels, values)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_title(f"Intent-parser accuracy by criterion (n={len(intent_rows)})")
        for i, v in enumerate(values):
            ax.text(min(v + 0.02, 0.97), i, f"{v:.0%}", va="center", fontsize=9)
        fig.tight_layout()
        p = FIGURES_DIR / "intent_accuracy.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths["intent_accuracy"] = str(p.relative_to(ANALYSIS_DIR.parent))

    # 2. Diversity comparison: paired bars per prompt.
    if diversity_rows:
        labels = [r.prompt[:40] + ("…" if len(r.prompt) > 40 else "") for r in diversity_rows]
        pop_e = [r.pop_rank_entropy for r in diversity_rows]
        llm_e = [r.llm_rank_entropy for r in diversity_rows]
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 2))
        bar_w = 0.4
        ax.barh([i + bar_w / 2 for i in x], pop_e, height=bar_w, label="popularity rank")
        ax.barh([i - bar_w / 2 for i in x], llm_e, height=bar_w, label="LLM rerank")
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Genre-level Shannon entropy (nats)")
        ax.set_title("Recommendation diversity: popularity vs LLM rerank")
        ax.legend(loc="lower right")
        fig.tight_layout()
        p = FIGURES_DIR / "diversity_comparison.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths["diversity_comparison"] = str(p.relative_to(ANALYSIS_DIR.parent))

    # 3. Latency stacked bar per prompt.
    if diversity_rows:
        labels = [r.prompt[:30] + ("…" if len(r.prompt) > 30 else "") for r in diversity_rows]
        parse = [r.parse_ms for r in diversity_rows]
        search = [r.search_ms for r in diversity_rows]
        rerank = [r.rerank_ms for r in diversity_rows]
        yt = [r.youtube_ms for r in diversity_rows]
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(labels) + 2))
        ax.barh(x, parse, label="LLM parse")
        ax.barh(x, search, left=parse, label="iTunes search")
        bot2 = [a + b for a, b in zip(parse, search)]
        ax.barh(x, rerank, left=bot2, label="LLM rerank")
        bot3 = [a + b for a, b in zip(bot2, rerank)]
        ax.barh(x, yt, left=bot3, label="YouTube lookup")
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Time (ms)")
        ax.set_title("End-to-end latency breakdown")
        ax.legend(loc="lower right")
        fig.tight_layout()
        p = FIGURES_DIR / "latency_breakdown.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths["latency_breakdown"] = str(p.relative_to(ANALYSIS_DIR.parent))

    return paths


# --------------------------------- main ---------------------------------- #


@dataclass
class EvalSummary:
    intent_n: int
    intent_overall_accuracy: float
    intent_per_criterion: dict[str, float] = field(default_factory=dict)
    intent_mean_parse_ms: float = 0.0
    diversity_n: int = 0
    mean_pop_entropy: float = 0.0
    mean_llm_entropy: float = 0.0
    mean_pop_entropy_diverse: float = 0.0
    mean_llm_entropy_diverse: float = 0.0
    mean_total_ms: float = 0.0
    figures: dict[str, str] = field(default_factory=dict)


def main() -> None:
    load_dotenv()
    print("== Building agent…")
    llm = GeminiLLM()
    music = ItunesClient()
    youtube = YouTubeClient()
    agent = RecommenderAgent(llm=llm, music=music, youtube=youtube)

    print(f"== Experiment 1: intent parsing on {len(LABELED)} prompts…")
    intent_rows = run_intent_eval(llm)
    overall_pass_rate = (
        sum(r.overall_pass for r in intent_rows) / max(1, len(intent_rows))
    )
    crit_totals: dict[str, list[bool]] = {}
    for r in intent_rows:
        for k, v in r.passed_criteria.items():
            crit_totals.setdefault(k, []).append(v)
    per_criterion = {
        k: round(sum(v) / len(v), 4) for k, v in crit_totals.items()
    }
    print(f"   overall accuracy = {overall_pass_rate:.1%}")

    print(f"== Experiment 2: diversity ablation on {len(DIVERSITY_PROMPTS)} prompts…")
    diversity_rows = run_diversity_eval(agent)

    diverse_only = [r for r in diversity_rows if r.requested_diverse]
    summary = EvalSummary(
        intent_n=len(intent_rows),
        intent_overall_accuracy=round(overall_pass_rate, 4),
        intent_per_criterion=per_criterion,
        intent_mean_parse_ms=round(
            sum(r.parse_ms for r in intent_rows) / max(1, len(intent_rows)), 1
        ),
        diversity_n=len(diversity_rows),
        mean_pop_entropy=round(
            sum(r.pop_rank_entropy for r in diversity_rows) / max(1, len(diversity_rows)), 3
        ),
        mean_llm_entropy=round(
            sum(r.llm_rank_entropy for r in diversity_rows) / max(1, len(diversity_rows)), 3
        ),
        mean_pop_entropy_diverse=round(
            sum(r.pop_rank_entropy for r in diverse_only) / max(1, len(diverse_only)), 3
        )
        if diverse_only
        else 0.0,
        mean_llm_entropy_diverse=round(
            sum(r.llm_rank_entropy for r in diverse_only) / max(1, len(diverse_only)), 3
        )
        if diverse_only
        else 0.0,
        mean_total_ms=round(
            sum(r.total_ms for r in diversity_rows) / max(1, len(diversity_rows)), 1
        ),
    )

    print("== Rendering plots…")
    summary.figures = make_plots(intent_rows, diversity_rows)

    out = {
        "summary": asdict(summary),
        "intent_rows": [asdict(r) for r in intent_rows],
        "diversity_rows": [asdict(r) for r in diversity_rows],
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"== Wrote {RESULTS_PATH}")
    print(f"== Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
