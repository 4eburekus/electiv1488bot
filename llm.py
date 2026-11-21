"""Подключение к модели: один интерфейс для любого провайдера.

Все провайдеры ниже говорят на одном «диалекте» — OpenAI-совместимом API (Chat Completions
с tools). Поэтому клиент один, меняются только адрес, ключ и модель. Выбор — в файле .env:

    LLM_PROVIDER=mock         учебная имитация, без сети и ключей (по умолчанию)
    LLM_PROVIDER=openrouter   как в common_ai_agents; нужен OPENROUTER_API_KEY
    LLM_PROVIDER=ollama       модель на вашем компьютере, http://localhost:11434/v1
    LLM_PROVIDER=lmstudio     модель на вашем компьютере, http://localhost:1234/v1
    LLM_PROVIDER=custom       любой OpenAI-совместимый сервис: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

Подробности и примеры адресов — в README.md.
"""
import json
import os
import time
from datetime import date, timedelta

from dotenv import load_dotenv

from bot.mock_llm import MockLLM

DEFAULTS = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
# Модели OpenRouter: порядок проверки. Список бесплатных моделей меняется, правила отбора остаются.
PREFERRED_OPENROUTER = ["google/gemma-4-31b-it:free", "nvidia/nemotron-3-super-120b-a12b:free"]
FALLBACK_OPENROUTER = "openrouter/free"


class LLMConfigError(RuntimeError):
    """Провайдер настроен неверно; сообщение говорит, что исправить."""


def _is_suitable(m: dict, min_context: int = 32_000, min_days_left: int = 30) -> bool:
    params = m.get("supported_parameters") or []
    outputs = (m.get("architecture") or {}).get("output_modalities") or []
    expires = m.get("expiration_date")
    expiring = bool(expires) and date.fromisoformat(expires) < date.today() + timedelta(days=min_days_left)
    specialised = any(w in m["id"] for w in ("code", "safety", "guard"))
    return ("tools" in params and outputs == ["text"] and not specialised
            and (m.get("context_length") or 0) >= min_context and not expiring)


def pick_openrouter_model(models: list) -> str:
    """Свободная модель с поддержкой tools. Явный выбор — OPENROUTER_MODEL в .env."""
    if os.getenv("OPENROUTER_MODEL"):
        return os.environ["OPENROUTER_MODEL"]
    free = [m for m in models if (m.get("pricing") or {}).get("prompt") == "0"
            and (m.get("pricing") or {}).get("completion") == "0" and _is_suitable(m)]
    ids = {m["id"] for m in free}
    for pref in PREFERRED_OPENROUTER:
        if pref in ids:
            return pref
    return sorted(ids)[0] if ids else FALLBACK_OPENROUTER


class OpenAICompatLLM:
    """chat(messages, tools) -> {"content", "tool_calls"}: тот же формат, что у MockLLM."""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or "none", timeout=120)
        self.model = model
        self.name = f"{base_url} · {model}"

    def chat(self, messages: list, tools: list | None = None) -> dict:
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools or None)
                break
            except Exception as exc:  # лимиты бесплатных моделей: подождать и повторить
                if attempt == 3 or "429" not in str(exc):
                    raise
                time.sleep(5 * (attempt + 1))
        msg = r.choices[0].message
        calls = [{"id": c.id, "name": c.function.name, "arguments": json.loads(c.function.arguments or "{}")}
                 for c in (msg.tool_calls or [])]
        return {"content": msg.content, "tool_calls": calls}


def _first_local_model(base_url: str, api_key: str) -> str:
    from openai import OpenAI
    try:
        models = OpenAI(base_url=base_url, api_key=api_key or "none", timeout=10).models.list().data
    except Exception as exc:
        raise LLMConfigError(f"Не удалось подключиться к {base_url}: {exc}. Сервер запущен?") from exc
    if not models:
        raise LLMConfigError(f"На {base_url} нет загруженных моделей. Загрузите модель с поддержкой tools.")
    return models[0].id


def make_llm(provider: str | None = None):
    """Создаёт клиента по LLM_PROVIDER (или по аргументу)."""
    load_dotenv()
    provider = (provider or os.getenv("LLM_PROVIDER") or "mock").lower()
    if provider == "mock":
        return MockLLM()

    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise LLMConfigError("Нет OPENROUTER_API_KEY в .env. Ключ: https://openrouter.ai/keys. "
                                 "Если OpenRouter у вас недоступен — LLM_PROVIDER=ollama или custom (см. README).")
        import requests
        try:
            models = requests.get(f"{DEFAULTS['openrouter']}/models", timeout=10).json().get("data", [])
        except Exception:
            models = []
        return OpenAICompatLLM(DEFAULTS["openrouter"], key, pick_openrouter_model(models))

    if provider in ("ollama", "lmstudio"):
        base = os.getenv(f"{provider.upper()}_BASE_URL", DEFAULTS[provider])
        model = os.getenv("LLM_MODEL") or _first_local_model(base, "none")
        return OpenAICompatLLM(base, "none", model)

    if provider == "custom":
        base, model = os.getenv("LLM_BASE_URL"), os.getenv("LLM_MODEL")
        if not (base and model):
            raise LLMConfigError("Для custom нужны LLM_BASE_URL и LLM_MODEL (и LLM_API_KEY, если сервис требует ключ).")
        return OpenAICompatLLM(base, os.getenv("LLM_API_KEY", ""), model)

    raise LLMConfigError(f"Неизвестный LLM_PROVIDER={provider!r}. Допустимо: mock, openrouter, ollama, lmstudio, custom.")
