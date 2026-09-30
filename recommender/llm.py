"""LLM facade — structured intent parsing + candidate reranking.

Supports two providers behind one interface:

* **Groq** (default when `GROQ_API_KEY` is set). Free tier ~1000 RPD per
  model and very fast (sub-second). Uses gpt-oss-120b with JSON mode
  (Llama 3.3 70B until Groq decommissioned it on 2026-08-16).
* **Gemini** (fallback when only `GEMINI_API_KEY` is set, and the backup
  provider at call time when both keys are set). Free tier on fresh
  projects is only ~20 RPD as of April 2026, which is too tight for
  evaluation but fine for a low-traffic demo.

Both calls return JSON validated against a Pydantic schema, so the
agent never has to regex-extract fields from prose.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from recommender.prompts import (
    INTENT_SYSTEM_PROMPT,
    RERANK_SYSTEM_PROMPT,
    candidates_block,
    conversation_state_block,
)


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
#  Pydantic schemas — used both as response_schema and for runtime validation #
# --------------------------------------------------------------------------- #


class SearchQuery(BaseModel):
    mood_keywords: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    seed_artists: list[str] = Field(default_factory=list)
    new_exclude_artists: list[str] = Field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    count: int = 5
    diversify: bool = False
    more_of_same: bool = False
    acknowledge_only: bool = False
    assistant_intent_summary: str = ""


class RerankResult(BaseModel):
    chosen_track_ids: list[str] = Field(default_factory=list)
    message: str = ""


# --------------------------------------------------------------------------- #
#  Provider abstraction                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class LLMTimings:
    parse_ms: float = 0.0
    rerank_ms: float = 0.0


class _Provider:
    name: str = "abstract"

    def generate_json(
        self,
        system_prompt: str,
        user_payload: str,
        schema: type[BaseModel],
        temperature: float,
    ) -> dict:  # pragma: no cover
        raise NotImplementedError


# --------------------------- Groq ---------------------------------------- #


class _GroqProvider(_Provider):
    """Groq supports OpenAI-style JSON mode. We can't push a full JSON Schema
    through, so we paste the schema text into the system prompt and rely on
    `response_format={"type": "json_object"}` to keep the output valid JSON,
    then validate it locally with Pydantic."""

    name = "groq"

    DEFAULT_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-16

    def __init__(self, api_key: str, model: str | None = None) -> None:
        from groq import Groq  # local import to keep package import light

        self._client = Groq(api_key=api_key)
        self.model = model or os.environ.get("GROQ_MODEL", self.DEFAULT_MODEL)

    def generate_json(
        self,
        system_prompt: str,
        user_payload: str,
        schema: type[BaseModel],
        temperature: float,
    ) -> dict:
        schema_text = json.dumps(schema.model_json_schema(), indent=2)
        sys = (
            system_prompt
            + "\n\nThe response MUST be a single JSON object that conforms to "
            + "this schema. Do NOT wrap it in markdown fences.\n\nSchema:\n"
            + schema_text
        )
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user_payload},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    max_completion_tokens=2048,
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise LLMError("Empty response from Groq.")
                return json.loads(text)
            except json.JSONDecodeError as e:
                last_err = e
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                if "rate" in msg or "429" in msg:
                    time.sleep(2 ** attempt + 1)
                    continue
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
        raise LLMError(f"Groq call failed: {last_err}")


# -------------------------- Gemini --------------------------------------- #


class _GeminiProvider(_Provider):
    name = "gemini"

    DEFAULT_MODEL = "gemini-2.5-flash-lite"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", self.DEFAULT_MODEL)

    @staticmethod
    def _extract_retry_seconds(err: Exception) -> float | None:
        m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(err))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def generate_json(
        self,
        system_prompt: str,
        user_payload: str,
        schema: type[BaseModel],
        temperature: float,
    ) -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=user_payload, config=config
                )
                text = (resp.text or "").strip()
                if not text:
                    raise LLMError("Empty response from Gemini.")
                return json.loads(text)
            except json.JSONDecodeError as e:
                last_err = e
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    wait = self._extract_retry_seconds(e) or (2 ** attempt) * 5
                    if wait > 60:
                        break
                    time.sleep(wait)
                    continue
            if attempt < 2:
                time.sleep(0.6 * (attempt + 1))
        raise LLMError(f"Gemini call failed: {last_err}")


# --------------------------------------------------------------------------- #
#  Public facade                                                              #
# --------------------------------------------------------------------------- #


class GeminiLLM:
    """Historical name (kept for API stability). Picks a provider at
    construction time:

      * If `GROQ_API_KEY` is set in the env, use Groq (recommended — much
        higher free-tier quota than Gemini).
      * Otherwise, fall back to Gemini.
      * `LLM_PROVIDER` env var ("groq" / "gemini") forces a specific choice.

    When both keys are set, Gemini also acts as a call-time backup: if the
    Groq call fails (outage, decommissioned model, exhausted quota), the same
    request is retried once on Gemini.

    The attribute name `model` reports whichever model is in use, prefixed
    with the provider for clarity ("groq:openai/gpt-oss-120b").
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        chosen = (provider or os.environ.get("LLM_PROVIDER") or "").strip().lower()
        groq_key = api_key if chosen == "groq" else os.environ.get("GROQ_API_KEY", "")
        gemini_key = api_key if chosen == "gemini" else os.environ.get("GEMINI_API_KEY", "")
        if not chosen:
            chosen = "groq" if groq_key else "gemini"
        if chosen == "groq":
            if not groq_key:
                raise LLMError(
                    "GROQ_API_KEY is not set (and you asked for the groq provider)."
                )
            self._provider = _GroqProvider(api_key=groq_key, model=model)
        elif chosen == "gemini":
            if not gemini_key:
                raise LLMError(
                    "GEMINI_API_KEY is not set (and you asked for the gemini provider)."
                )
            self._provider = _GeminiProvider(api_key=gemini_key, model=model)
        else:
            raise LLMError(f"Unknown LLM_PROVIDER {chosen!r}")
        self._fallback: _Provider | None = None
        if chosen == "groq" and gemini_key:
            self._fallback = _GeminiProvider(api_key=gemini_key)
        self.timings = LLMTimings()

    def _generate_json(
        self,
        system_prompt: str,
        user_payload: str,
        schema: type[BaseModel],
        temperature: float,
    ) -> dict:
        try:
            return self._provider.generate_json(
                system_prompt, user_payload, schema, temperature
            )
        except LLMError as primary_err:
            if self._fallback is None:
                raise
            try:
                return self._fallback.generate_json(
                    system_prompt, user_payload, schema, temperature
                )
            except LLMError as fallback_err:
                raise LLMError(
                    f"{primary_err}; fallback also failed: {fallback_err}"
                ) from fallback_err

    @property
    def model(self) -> str:
        return f"{self._provider.name}:{self._provider.model}"

    def parse_intent(
        self,
        user_text: str,
        history_summary: str,
        must_exclude_artists: list[str],
        liked_artists: list[str],
        recent_genres: list[str],
    ) -> SearchQuery:
        payload = (
            conversation_state_block(
                history_summary, must_exclude_artists, liked_artists, recent_genres
            )
            + "\n\n[User]\n"
            + user_text.strip()
        )
        t0 = time.perf_counter()
        try:
            raw = self._generate_json(
                INTENT_SYSTEM_PROMPT, payload, SearchQuery, temperature=0.2
            )
            obj = SearchQuery.model_validate(raw)
        finally:
            self.timings.parse_ms = (time.perf_counter() - t0) * 1000
        return obj

    def rerank(
        self,
        user_text: str,
        history_summary: str,
        must_exclude_artists: list[str],
        candidates: list[dict],
        target_count: int,
        diversify: bool,
    ) -> RerankResult:
        payload = (
            conversation_state_block(
                history_summary, must_exclude_artists, [], []
            )
            + f"\n\ntarget_count: {target_count}\n"
            + f"diversify: {str(diversify).lower()}\n\n"
            + candidates_block(candidates)
            + "\n\n[User]\n"
            + user_text.strip()
        )
        t0 = time.perf_counter()
        try:
            raw = self._generate_json(
                RERANK_SYSTEM_PROMPT, payload, RerankResult, temperature=0.4
            )
            obj = RerankResult.model_validate(raw)
        finally:
            self.timings.rerank_ms = (time.perf_counter() - t0) * 1000
        return obj
