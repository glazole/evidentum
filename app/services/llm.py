from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI


# ── Yandex Foundation Models config ─────────────────────────────────────────────
YC_OPENAI_BASE_URL = os.getenv("YC_OPENAI_BASE_URL", "https://llm.api.cloud.yandex.net/v1")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

# ── OpenAI (ChatGPT) config ──────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ── Anthropic (Claude) config ────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Shared defaults ──────────────────────────────────────────────────────────────
# LLM_PROVIDER selects which backend to use. Supported: yandex | openai | anthropic
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "yandex")
LLM_MODEL_FAMILY = os.getenv("LLM_MODEL_FAMILY", "yandexgpt")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))


# ── Model alias tables ───────────────────────────────────────────────────────────
# Each table maps short alias → provider-native model name.
# Yandex-style aliases ("alice", "lite", …) are mirrored in OpenAI/Anthropic tables
# so switching LLM_PROVIDER requires no prompt or code changes.

YANDEX_MODEL_ALIASES: dict[str, str] = {
    "yandex": "yandexgpt-5.1",
    "yandexgpt": "yandexgpt-5.1",
    "pro": "yandexgpt-5.1",
    "alice": "aliceai-llm",
    "aliceai": "aliceai-llm",
    "lite": "yandexgpt-lite",
}

OPENAI_MODEL_ALIASES: dict[str, str] = {
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt4": "gpt-4-turbo",
    "gpt4turbo": "gpt-4-turbo",
    "mini": "gpt-4o-mini",
    "gpt4omini": "gpt-4o-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    # Yandex-style aliases → equivalent OpenAI models
    "yandex": "gpt-4o",
    "yandexgpt": "gpt-4o",
    "pro": "gpt-4o",
    "alice": "gpt-4o",
    "lite": "gpt-4o-mini",
}

ANTHROPIC_MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-5-20251101",
    "claude-opus": "claude-opus-4-5-20251101",
    "sonnet": "claude-sonnet-4-5-20251101",
    "claude-sonnet": "claude-sonnet-4-5-20251101",
    "haiku": "claude-haiku-3-5-20251022",
    "claude-haiku": "claude-haiku-3-5-20251022",
    # Yandex-style aliases → equivalent Anthropic models
    "yandex": "claude-sonnet-4-5-20251101",
    "yandexgpt": "claude-sonnet-4-5-20251101",
    "pro": "claude-sonnet-4-5-20251101",
    "alice": "claude-opus-4-5-20251101",
    "lite": "claude-haiku-3-5-20251022",
}


# ── Abstract base ────────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """
    Provider-agnostic interface for text completion.

    Concrete implementations:
      - YandexLLMClient  (default, uses Yandex Foundation Models)
      - OpenAILLMClient  (ChatGPT, set OPENAI_API_KEY)
      - AnthropicLLMClient (Claude, set ANTHROPIC_API_KEY + pip install anthropic)

    Use get_llm_client() factory to select provider via LLM_PROVIDER env var.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send messages to the LLM and return the assistant reply as plain text."""

    @abstractmethod
    def resolve_model(self, model: str | None = None) -> str:
        """Return the canonical model identifier (URI or name) for display/logging."""

    def complete_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Call the LLM and parse the response as JSON.

        Tries three strategies in order:
          1. Direct json.loads on the raw response.
          2. Strip markdown fences (```json ... ```) then parse.
          3. Regex-extract the first {...} block then parse.
        Raises ValueError if all strategies fail.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        raw = self.complete(messages, model=model, temperature=temperature, max_tokens=max_tokens)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:500]}")


# ── Yandex Foundation Models ─────────────────────────────────────────────────────

class YandexLLMClient(BaseLLMClient):
    """Thin wrapper around the Yandex Foundation Models OpenAI-compatible endpoint."""

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

    def resolve_model(self, model: str | None = None) -> str:
        family = (model or self.model_family).strip().lower()
        model_id = YANDEX_MODEL_ALIASES.get(family, family)
        return f"gpt://{self.folder_id}/{model_id}/latest"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        model_uri = self.resolve_model(model)
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model_uri,
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


# ── OpenAI (ChatGPT) ─────────────────────────────────────────────────────────────

class OpenAILLMClient(BaseLLMClient):
    """
    OpenAI ChatGPT client.

    Required .env var: OPENAI_API_KEY
    Optional .env var: OPENAI_BASE_URL (useful for Azure OpenAI or LM Studio)
    Optional .env var: LLM_MODEL_FAMILY — default model alias (e.g. "gpt4o", "mini")

    Yandex-style aliases ("alice", "pro", "lite") are automatically mapped to
    equivalent OpenAI models so LLM_PROVIDER=openai works without other changes.
    """

    def __init__(
        self,
        *,
        api_key: str = OPENAI_API_KEY,
        base_url: str = OPENAI_BASE_URL,
        model_family: str = LLM_MODEL_FAMILY,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        debug: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to .env or set LLM_PROVIDER=yandex to keep using Yandex."
            )

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

    def resolve_model(self, model: str | None = None) -> str:
        family = (model or self.model_family).strip().lower()
        return OPENAI_MODEL_ALIASES.get(family, family)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        model_name = self.resolve_model(model)
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model_name,
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
                    print(f"OpenAI attempt {attempt} failed: {exc}. Retrying in {delay}s…")
                time.sleep(delay)
        raise RuntimeError(f"OpenAI failed after {self.max_retries} attempts: {last_exc}") from last_exc


# ── Anthropic (Claude) ───────────────────────────────────────────────────────────

class AnthropicLLMClient(BaseLLMClient):
    """
    Anthropic Claude client.

    Required: pip install anthropic
    Required .env var: ANTHROPIC_API_KEY
    Optional .env var: LLM_MODEL_FAMILY — default alias (e.g. "sonnet", "haiku", "opus")

    Yandex-style aliases ("alice", "pro", "lite") are automatically mapped to
    equivalent Claude models so LLM_PROVIDER=anthropic works without other changes.

    Note: Anthropic uses a separate `system` parameter instead of a system role
    in the messages array. This is handled transparently in complete().
    """

    def __init__(
        self,
        *,
        api_key: str = ANTHROPIC_API_KEY,
        model_family: str = LLM_MODEL_FAMILY,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        debug: bool = False,
    ) -> None:
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic
        except ImportError as exc:
            raise ImportError(
                "Package 'anthropic' is required for AnthropicLLMClient. "
                "Install it with: pip install anthropic"
            ) from exc

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to .env or set LLM_PROVIDER=yandex to keep using Yandex."
            )

        self.model_family = model_family
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.debug = debug

        self._client = _anthropic.Anthropic(api_key=api_key)

    def resolve_model(self, model: str | None = None) -> str:
        family = (model or self.model_family).strip().lower()
        return ANTHROPIC_MODEL_ALIASES.get(family, family)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        model_name = self.resolve_model(model)
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Anthropic separates system instructions from the human/assistant turns
        system_text: str | None = None
        anthropic_messages: list[dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            else:
                anthropic_messages.append({"role": role, "content": content})

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = dict(
                    model=model_name,
                    max_tokens=tokens,
                    messages=anthropic_messages,
                )
                if system_text:
                    kwargs["system"] = system_text
                if temp is not None:
                    kwargs["temperature"] = temp

                response = self._client.messages.create(**kwargs)
                text = response.content[0].text if response.content else ""
                if self.debug:
                    print("LLM response:", text[:300])
                return text.strip()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                if self.debug:
                    print(f"Anthropic attempt {attempt} failed: {exc}. Retrying in {delay}s…")
                time.sleep(delay)
        raise RuntimeError(f"Anthropic failed after {self.max_retries} attempts: {last_exc}") from last_exc


# ── Factory ──────────────────────────────────────────────────────────────────────

def get_llm_client(
    provider: str | None = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Create an LLM client for the configured provider.

    Provider resolution order:
      1. ``provider`` argument
      2. LLM_PROVIDER env var   (default: "yandex")

    Supported providers:
      "yandex"    — Yandex Foundation Models (requires YANDEX_FOLDER_ID + YANDEX_API_KEY)
      "openai"    — OpenAI ChatGPT           (requires OPENAI_API_KEY)
      "anthropic" — Anthropic Claude         (requires ANTHROPIC_API_KEY + pip install anthropic)

    Extra keyword arguments are forwarded to the client constructor
    (e.g. debug=True, temperature=0.1).

    Examples::

        # Use whatever provider is set in .env
        llm = get_llm_client()

        # Force a specific provider
        llm = get_llm_client("openai")

        # Override model for this call
        llm = get_llm_client(model_family="gpt4o-mini")
    """
    p = (provider or LLM_PROVIDER or "yandex").strip().lower()
    if p in ("yandex", "yandexgpt", "yc"):
        return YandexLLMClient(**kwargs)
    if p in ("openai", "chatgpt", "gpt"):
        return OpenAILLMClient(**kwargs)
    if p in ("anthropic", "claude"):
        return AnthropicLLMClient(**kwargs)
    raise ValueError(
        f"Unknown LLM provider: {p!r}. "
        "Supported values: 'yandex', 'openai', 'anthropic'."
    )
