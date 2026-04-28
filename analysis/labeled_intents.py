"""Hand-labeled benchmark for the intent parser.

Each entry is (utterance, expected). The expected dict only lists the fields
we care about for the test; missing keys are not checked. This keeps the
labels stable when we add new SearchQuery fields later.
"""

from __future__ import annotations

LABELED: list[tuple[str, dict]] = [
    # ----- mood / vibe ------------------------------------------------- #
    ("Recommend me some sad songs.",
     {"mood_keywords_nonempty": True, "acknowledge_only": False}),
    ("Songs for a rainy nighttime drive.",
     {"mood_keywords_nonempty": True}),
    ("Five upbeat indie tracks please.",
     {"genres_contains_any": ["indie", "indie pop", "indie rock"], "count": 5}),
    ("I want music that fits a focused study session, no lyrics.",
     {"mood_keywords_nonempty": True}),
    ("Give me feel-good summer songs.",
     {"mood_keywords_nonempty": True}),
    ("Slow, melancholy piano ballads.",
     {"mood_keywords_nonempty": True}),

    # ----- genre / era -------------------------------------------------- #
    ("Classic 70s rock anthems.",
     {"year_min_lte": 1979, "year_max_gte": 1970}),
    ("90s grunge.",
     {"year_min_lte": 1995, "year_max_gte": 1990}),
    ("Synthwave from the 2010s.",
     {"genres_contains_any": ["synthwave", "synth-pop", "synth pop"]}),
    ("Old-school hip hop.",
     {"genres_contains_any": ["hip hop", "rap"]}),

    # ----- artist references ------------------------------------------- #
    ("Anything similar to Bon Iver but more electronic.",
     {"seed_artists_contains_any": ["bon iver"]}),
    ("Stuff that sounds like Phoebe Bridgers and Big Thief.",
     {"seed_artists_contains_any": ["phoebe bridgers", "big thief"]}),

    # ----- exclusions --------------------------------------------------- #
    ("Stop recommending Taylor Swift, please.",
     {"new_exclude_artists_contains_any": ["taylor swift"], "acknowledge_only": True}),
    ("No more Coldplay.",
     {"new_exclude_artists_contains_any": ["coldplay"], "acknowledge_only": True}),
    ("I'm tired of Drake.",
     {"new_exclude_artists_contains_any": ["drake"], "acknowledge_only": True}),
    ("Anything but Ed Sheeran tonight.",
     {"new_exclude_artists_contains_any": ["ed sheeran"]}),

    # ----- diversity ---------------------------------------------------- #
    ("Mix it up — give me something more diverse next time.",
     {"diversify": True}),
    ("Surprise me with variety.",
     {"diversify": True}),
    ("Different artists this time.",
     {"diversify": True}),

    # ----- continuation ------------------------------------------------- #
    ("Show me more like that.",
     {"more_of_same": True}),
    ("Next.",
     {"more_of_same": True}),

    # ----- explicit count ---------------------------------------------- #
    ("Recommend me 8 chill electronic tracks.",
     {"count": 8}),
    ("Just give me 3 songs.",
     {"count": 3}),

    # ----- compound requests ------------------------------------------- #
    ("Indie folk from 2018-2022, no Mumford & Sons please.",
     {"genres_contains_any": ["indie folk", "folk", "indie"],
      "year_min_lte": 2018, "year_max_gte": 2022,
      "new_exclude_artists_contains_any": ["mumford & sons", "mumford and sons"]}),
    ("Five jazz tracks for late-night reading.",
     {"genres_contains_any": ["jazz"], "count": 5}),
    ("Songs that fit a long road trip through California, more diverse.",
     {"diversify": True, "mood_keywords_nonempty": True}),

    # ----- short / ambiguous ------------------------------------------- #
    ("More.",
     {"more_of_same": True}),
    ("Something fun.",
     {"mood_keywords_nonempty": True}),
    ("Energy.",
     {"mood_keywords_nonempty": True}),

    # ----- chill / ambient -------------------------------------------- #
    ("Lo-fi background music.",
     {"mood_keywords_nonempty": True}),
]


def n_labeled() -> int:
    return len(LABELED)
