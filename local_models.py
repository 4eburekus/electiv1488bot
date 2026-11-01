"""Локальные модели для раздела 11: LLM, эмбеддинги, реранкер.

Всё работает без облачных ключей: LLM и эмбеддинги берутся из локального
OpenAI-совместимого сервера (LM Studio / gpustack), реранкер — из HF-кэша.

Запуск проверки окружения:
    myvenv\\Scripts\\python.exe -X utf8 local_models.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Значения по умолчанию под локальное окружение ─────────────────────────────
DEFAULT_BASE_URL = "http://localhost:1235/v1"
DEFAULT_LLM_MODEL = "gemma-3-4b-it"
DEFAULT_EMBED_MODEL = "text-embedding-bge-m3"
DEFAULT_RERANKER = "BAAI/bge-reranker-base"


def base_url() -> str:
    return (os.getenv("OPENAI_API_BASE") or DEFAULT_BASE_URL).rstrip("/") + "/"


def llm_model_name() -> str:
    return os.getenv("OPENAI_MODEL") or DEFAULT_LLM_MODEL


def embed_model_name() -> str:
    return os.getenv("EMBED_MODEL") or DEFAULT_EMBED_MODEL


def _api_key() -> str:
    # Локальный сервер ключ не проверяет, но openai SDK требует непустое значение.
    return os.getenv("OPENAI_API_KEY") or "sk-local"


# ── LLM ───────────────────────────────────────────────────────────────────────
def build_llm(temperature: float = 0.2, max_tokens: int | None = None) -> ChatOpenAI:
    """Чат-модель с локального эндпоинта. Ключа облака не нужно."""
    kwargs: dict[str, Any] = {
        "model": llm_model_name(),
        "openai_api_key": _api_key(),
        "openai_api_base": base_url(),
        "temperature": temperature,
        "max_retries": 2,
        "timeout": float(os.getenv("LLM_TIMEOUT", "120")),
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def list_models() -> list[str]:
    """Список моделей, которые реально загружены/доступны на локальном сервере."""
    from openai import OpenAI

    client = OpenAI(api_key=_api_key(), base_url=base_url(), timeout=10, max_retries=0)
    return sorted(m.id for m in client.models.list().data)


# ── Эмбеддинги ────────────────────────────────────────────────────────────────
@dataclass
class LocalEmbeddings:
    """Эмбеддинги через локальный OpenAI-совместимый /embeddings (LM Studio, gpustack)."""

    model: str
    api_key: str = field(default_factory=_api_key)
    base_url: str = field(default_factory=base_url)

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120, max_retries=2)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        client = self._client()
        # батчами, чтобы не ловить таймаут на длинном списке
        step = 16
        for i in range(0, len(texts), step):
            res = client.embeddings.create(model=self.model, input=texts[i : i + step])
            out.extend(item.embedding for item in res.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings():
    """Выбор источника эмбеддингов: локальный API или sentence-transformers."""
    source = (os.getenv("EMBED_SOURCE") or "local").strip().lower()
    if source in ("local", "lmstudio", "gpustack", "openai"):
        return LocalEmbeddings(model=embed_model_name())
    if source in ("sentence_transformers", "st", "hf"):
        from langchain_huggingface import HuggingFaceEmbeddings

        model_id = os.getenv("HF_EMBED_MODEL") or "BAAI/bge-m3"
        device = os.getenv("HF_DEVICE") or "cpu"
        return HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    raise ValueError(f"Неизвестный EMBED_SOURCE={source!r}")


# ── Реранкер (раздел 6, локально) ─────────────────────────────────────────────
_reranker_cache: dict[str, Any] = {"model": None, "tried": False}


def get_reranker():
    """CrossEncoder из локального кэша; None, если модели нет или она отключена."""
    name = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER)
    if not name.strip():
        return None
    if _reranker_cache["tried"]:
        return _reranker_cache["model"]

    _reranker_cache["tried"] = True
    try:
        from sentence_transformers import CrossEncoder

        device = os.getenv("HF_DEVICE") or "cpu"
        _reranker_cache["model"] = CrossEncoder(name, device=device)
    except Exception as exc:  # модель не скачана — работаем без реранкинга
        print(f"⚠️ Реранкер недоступен ({type(exc).__name__}): поиск будет только по векторам")
        _reranker_cache["model"] = None
    return _reranker_cache["model"]


def rerank(query: str, documents: list[str], top_n: int = 5) -> list[tuple[int, float]]:
    """Возвращает [(индекс документа, скор)] в порядке убывания релевантности."""
    model = get_reranker()
    if not model or not documents:
        return [(i, 0.0) for i in range(min(top_n, len(documents)))]
    scores = model.predict([(query, doc) for doc in documents])
    order = sorted(range(len(documents)), key=lambda i: float(scores[i]), reverse=True)
    return [(i, float(scores[i])) for i in order[:top_n]]


# ── Проверка окружения ────────────────────────────────────────────────────────
def check_environment() -> bool:
    """Диагностика: какие локальные модели реально отвечают. Ничего не индексирует."""
    ok = True
    print(f"Эндпоинт: {base_url()}")
    try:
        models = list_models()
        print(f"✅ Моделей на сервере: {len(models)}")
        print(json.dumps(models, ensure_ascii=False, indent=2))
    except Exception as exc:
        ok = False
        print(f"❌ Локальный сервер недоступен: {type(exc).__name__}: {exc}")
        return ok

    want = llm_model_name()
    if want in models:
        print(f"✅ LLM '{want}' доступна")
    else:
        ok = False
        print(f"⚠️ LLM '{want}' нет в списке — укажите OPENAI_MODEL из списка выше")

    probe = build_llm(max_tokens=32)
    try:
        reply = probe.invoke("Ответь одним словом: сколько будет 2+2?")
        print(f"✅ LLM отвечает: {reply.content!r}")
    except Exception as exc:
        ok = False
        print(f"❌ LLM не отвечает: {type(exc).__name__}: {str(exc)[:200]}")

    emb = build_embeddings()
    try:
        vec = emb.embed_query("тест эмбеддинга")
        print(f"✅ Эмбеддинги: размерность {len(vec)}")
    except Exception as exc:
        ok = False
        print(f"❌ Эмбеддинги недоступны: {type(exc).__name__}: {str(exc)[:200]}")

    rr = get_reranker()
    if rr is None:
        print("ℹ️ Реранкер не используется")
    else:
        pairs = [("Что такое RAG?", "RAG — генерация с дополненной выборкой."), ("Что такое RAG?", "Я люблю пиццу.")]
        print(f"✅ Реранкер скоры: {[round(float(s), 3) for s in rr.predict(pairs)]}")

    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check_environment() else 1)
