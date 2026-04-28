"""Music chat recommender package.

Submodules are imported lazily so that exploring one piece of the system
(e.g. running `python -m recommender.spotify_client`) doesn't pull in
gradio / google-genai unnecessarily.
"""

__all__ = ["RecommenderAgent", "ConversationState"]


def __getattr__(name):  # PEP 562 lazy import
    if name == "RecommenderAgent":
        from recommender.agent import RecommenderAgent
        return RecommenderAgent
    if name == "ConversationState":
        from recommender.conversation import ConversationState
        return ConversationState
    raise AttributeError(f"module 'recommender' has no attribute {name!r}")
