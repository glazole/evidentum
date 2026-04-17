from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import OpenAI


YC_OPENAI_BASE_URL = os.getenv("YC_OPENAI_BASE_URL", "https://llm.api.cloud.yandex.net/v1")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

LLM_MODEL_FAMILY = os.getenv("LLM_MODEL_FAMILY", "yandexgpt")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

MODEL_ALIASES = {
    "yandex": "yandexgpt-5.1",
    "yandexgpt": "yandexgpt-5.1",
    "pro": "yandexgpt-5.1",
    "alice": "aliceai-llm",
    "aliceai": "aliceai-llm",
    "lite": "yandexgpt-lite",
}


class YandexLLMClient:
    """
    Thin wrapper around the Yandex Foundation Models OpenAI-compatible endpoint.

    Uses the same openai package and credentials as the main answer service.
    Supports JSON-mode extraction for structured enrichment tasks.
    """

    def __init__(
        self,
        *,
        folder_id: str = YANDEX_FOLDER_ID,
        api_key: str = YANDEX_API_KEY,
        base_url: str = YC_OPENAI_BASE_URL,
        model_family: str = LLM_MODEL_FAMILY,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        debug: bool = False,
    ) -> None:
        if not folder_id:
            raise ValueError("YANDEX_FOLDER_ID is not set.")
        if not api_key:
            raise ValueError("YANDEX_API_KEY is not set.")

        self.folder_id = folder_id
        self.model_family = model_family
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.debug = debug

        self._client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
        )

    def _model_uri(self, model_family: str | None = None) -> str:
        family = (model_family or self.model_family).strip().lower()
        model_id = MODEL_ALIASES.get(family, family)
        return f"gpt://{self.folder_id}/{model_id}/latest"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model_family: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call YandexGPT with retry on transient errors, return assistant text."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_uri(model_family),
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                    messages=messages,  # type: ignore[arg-type]
                )
                text = response.choices[0].message.content or ""
                if self.debug:
                    print("LLM response:", text[:300])
                return text.strip()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                if self.debug:
                    print(f"LLM attempt {attempt} failed: {exc}. Retrying in {delay}s…")
                time.sleep(delay)
        raise RuntimeError(f"YandexLLM failed after {self.max_retries} attempts: {last_exc}") from last_exc

    def complete_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model_family: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Call YandexGPT and parse response as JSON.

        Falls back to regex extraction of the first {...} block if the model
        wraps the JSON in markdown code fences or prose.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        raw = self.complete(messages, model_family=model_family, temperature=temperature, max_tokens=max_tokens)

        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strip markdown fences
        stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Extract first {...} block
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:500]}")
