from __future__ import annotations

import json
import os
from dataclasses import dataclass


DEFAULT_OPENAI_COMPATIBLE_MODEL = "openai/gpt-4o-mini"


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    model_name: str
    api_key: str | None
    api_base: str | None
    extra_headers: dict[str, str] | None = None

    def to_dspy_lm_kwargs(self, *, max_tokens: int, temperature: float) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": _litellm_openai_compatible_model_name(self.model_name),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": 180.0,
        }
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        return kwargs


def resolve_openai_compatible_config(
    model_name: str | None = None,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        model_name=model_name or DEFAULT_OPENAI_COMPATIBLE_MODEL,
        api_key=api_key if api_key is not None else os.environ.get("OPENAI_API_KEY"),
        api_base=api_base if api_base is not None else os.environ.get("OPENAI_BASE_URL"),
        extra_headers=extra_headers if extra_headers is not None else _extra_headers_from_env(),
    )


def require_openai_compatible_config(
    model_name: str | None = None,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> OpenAICompatibleConfig:
    config = resolve_openai_compatible_config(
        model_name,
        api_key=api_key,
        api_base=api_base,
        extra_headers=extra_headers,
    )
    if not config.api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY before running the DSPy RLM or GEPA paths. "
            "Use OPENAI_BASE_URL for non-default OpenAI-compatible endpoints."
        )
    return config


def _litellm_openai_compatible_model_name(model_name: str) -> str:
    if model_name.startswith("openai/"):
        return model_name
    return f"openai/{model_name}"


def _extra_headers_from_env() -> dict[str, str] | None:
    raw = os.environ.get("OPENAI_EXTRA_HEADERS_JSON")
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("OPENAI_EXTRA_HEADERS_JSON must be a JSON object")
    return {str(key): str(value) for key, value in data.items()}
