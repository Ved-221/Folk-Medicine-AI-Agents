"""Unified Gemini LLM Client Wrapper.

Enforces:
1. Single gateway for all Gemini API calls (generate_json and embed).
2. Disk caching keyed by (prompt_version, input_hash) to protect free-tier quotas.
3. Call-count guard (max_calls_per_run) to prevent runaway loops.
4. Exponential backoff retry on transient errors and rate limits.
5. Strict JSON parsing with a single retry before raising LLMJSONParseError.
6. Zero API key logging or printing.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors

from src.common.config import (
    CACHE_DIR,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
    MAX_LLM_CALLS_PER_RUN,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM operations."""


class MaxLLMCallsExceededError(LLMError):
    """Raised when the maximum allowed API calls in a run is exceeded."""


class LLMJSONParseError(LLMError):
    """Raised when the model response cannot be parsed as valid JSON."""


class LLMClient:
    """Singleton-style or instantiable client wrapper for Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = GEMINI_MODEL,
        embedding_model_name: str = GEMINI_EMBEDDING_MODEL,
        cache_dir: Path = CACHE_DIR,
        max_calls_per_run: int = MAX_LLM_CALLS_PER_RUN,
    ) -> None:
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.cache_dir = Path(cache_dir)
        self.max_calls_per_run = max_calls_per_run
        self.call_count = 0

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: genai.Client | None = None

    def _get_raw_client(self) -> genai.Client:
        """Initializes raw google-genai Client lazily."""
        if self._client is None:
            if not self.api_key:
                raise LLMError("GEMINI_API_KEY is not set in environment or config.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def reset_call_count(self) -> None:
        """Resets the live API call counter for testing or new runs."""
        self.call_count = 0

    def _get_json_cache_path(
        self, prompt: str, schema: dict[str, Any] | None, prompt_version: str
    ) -> Path:
        """Computes deterministic disk cache filepath."""
        schema_repr = json.dumps(schema, sort_keys=True) if schema else ""
        content_key = f"{prompt_version}:{prompt}:{schema_repr}"
        key_hash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{prompt_version}_{key_hash}.json"

    def _get_embed_cache_path(self, text: str) -> Path:
        """Computes deterministic disk cache filepath for embeddings."""
        content_key = f"embed:{self.embedding_model_name}:{text}"
        key_hash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"embed_{key_hash}.json"

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        prompt_version: str = "v1",
        use_cache: bool = True,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Generates a structured JSON response from Gemini, backed by cache and retries.

        Args:
            prompt: Prompt instruction string.
            schema: Optional JSON schema dict for output constraint.
            prompt_version: Version identifier of the prompt for cache isolation.
            use_cache: Whether to read/write disk cache.
            max_retries: Number of parse/network retries.

        Returns:
            Parsed dictionary response.

        Raises:
            MaxLLMCallsExceededError: If call_count reaches max_calls_per_run.
            LLMJSONParseError: If model response cannot be parsed as JSON after retries.
            LLMError: On unrecoverable API errors.
        """
        cache_path = self._get_json_cache_path(prompt, schema, prompt_version)

        # 1. Check disk cache
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    logger.debug("Cache hit for prompt_version %s", prompt_version)
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed reading cache at %s: %s", cache_path, e)

        # 2. Check call guard
        if self.call_count >= self.max_calls_per_run:
            raise MaxLLMCallsExceededError(
                f"Call guard exceeded: maximum of {self.max_calls_per_run} calls per run reached."
            )

        client = self._get_raw_client()

        # Build config
        config: dict[str, Any] = {"response_mime_type": "application/json"}
        if schema:
            config["response_schema"] = schema

        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                time.sleep(2**attempt * 0.5)

            try:
                self.call_count += 1
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

                raw_text = response.text or ""
                # Parse JSON strictly
                parsed = json.loads(raw_text)

                # Save to disk cache
                if use_cache:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(parsed, f, ensure_ascii=False, indent=2)

                return parsed

            except json.JSONDecodeError as e:
                logger.warning("JSON decode failed on attempt %d: %s", attempt + 1, e)
                last_error = LLMJSONParseError(f"Model returned invalid JSON: {e}")
                if attempt == max_retries:
                    raise last_error from e

            except genai_errors.APIError as e:
                logger.warning("Gemini API error on attempt %d: %s", attempt + 1, e)
                last_error = LLMError(f"Gemini API failure: {e}")
                if attempt == max_retries:
                    raise last_error from e

            except Exception as e:
                logger.warning("Unexpected error during generate_json: %s", e)
                last_error = e
                if attempt == max_retries:
                    raise LLMError(f"Call failed: {e}") from e

        if last_error:
            raise last_error
        raise LLMError("Unknown failure in generate_json")

    def embed(
        self,
        texts: list[str],
        use_cache: bool = True,
    ) -> list[list[float]]:
        """Generates embedding vectors for a list of texts with individual disk caching.

        Args:
            texts: List of strings to embed.
            use_cache: Whether to read/write disk cache.

        Returns:
            List of float embedding vectors matching input order.
        """
        results: list[list[float] | None] = [None] * len(texts)
        texts_to_fetch_idx: list[int] = []

        # Check cache
        for idx, text in enumerate(texts):
            if not text.strip():
                # Zero vector for blank text
                results[idx] = [0.0] * 768
                continue

            cache_path = self._get_embed_cache_path(text)
            if use_cache and cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        results[idx] = json.load(f)
                    continue
                except Exception:
                    pass

            texts_to_fetch_idx.append(idx)

        if not texts_to_fetch_idx:
            return [res for res in results if res is not None]

        # Call API for uncached texts
        if self.call_count >= self.max_calls_per_run:
            raise MaxLLMCallsExceededError(
                f"Call guard exceeded: maximum of {self.max_calls_per_run} calls per run reached."
            )

        client = self._get_raw_client()

        for idx in texts_to_fetch_idx:
            text = texts[idx]
            self.call_count += 1
            response = client.models.embed_content(
                model=self.embedding_model_name,
                contents=text,
            )
            # Response embeddings extraction
            vec = response.embedding.values if hasattr(response, "embedding") and response.embedding else []
            if not vec and hasattr(response, "embeddings") and response.embeddings:
                vec = response.embeddings[0].values

            results[idx] = list(vec)

            if use_cache:
                cache_path = self._get_embed_cache_path(text)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(vec, f)

        return [res for res in results if res is not None]


# Global singleton instance
default_client = LLMClient()


def generate_json(
    prompt: str,
    schema: dict[str, Any] | None = None,
    prompt_version: str = "v1",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Convenience function calling the default client."""
    return default_client.generate_json(
        prompt=prompt,
        schema=schema,
        prompt_version=prompt_version,
        use_cache=use_cache,
    )


def embed(
    texts: list[str],
    use_cache: bool = True,
) -> list[list[float]]:
    """Convenience function calling the default client."""
    return default_client.embed(texts=texts, use_cache=use_cache)
