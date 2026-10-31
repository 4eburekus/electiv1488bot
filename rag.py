import math
import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.txt"
CACHE_PATH = BASE_DIR / ".rag_index.pkl"

# Локальный сервер эмбеддингов (bionic / llama.cpp, OpenAI-совместимый API).
# Переопределяется в .env: EMBED_BASE_URL / EMBED_MODEL / EMBED_API_KEY
EMBED_BASE = os.getenv("EMBED_BASE_URL", "http://localhost:1235/v1")
# id модели берём ровно тот, что отдаёт GET {EMBED_BASE}/models
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-bge-m3")
# Локальному серверу ключ не нужен, но openai-клиент требует непустую строку.
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or "local"


MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 200
SIMILARITY_THRESHOLD = 0.45
TOP_K = 3

_index = None


def _embeddings_client() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=EMBED_API_KEY,
        openai_api_base=EMBED_BASE,
        # локальные серверы не считают токены через tiktoken — отключаем,
        # иначе падает на подсчёте длины перед запросом
        check_embedding_ctx_length=False,
        # encoding_format — не поле OpenAIEmbeddings, поэтому через model_kwargs
        model_kwargs={"encoding_format": "float"},
    )


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _split_units(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _semantic_chunks(units: list[str], embeddings: OpenAIEmbeddings) -> list[str]:
    """Склеивает соседние абзацы, пока baai/bge-m3 считает их похожими."""
    if not units:
        return []
    unit_vecs = embeddings.embed_documents(units)
    chunks = []
    current = units[0]
    current_vec = unit_vecs[0]
    for text, vec in zip(units[1:], unit_vecs[1:]):
        merged = f"{current}\n\n{text}"
        similar = _cosine(current_vec, vec) >= SIMILARITY_THRESHOLD
        if similar and len(merged) <= MAX_CHUNK_CHARS:
            current = merged
            current_vec = [(c + v) / 2 for c, v in zip(current_vec, vec)]
            continue
        chunks.append(current)
        current = text
        current_vec = vec
    chunks.append(current)

    merged_chunks = []
    for chunk in chunks:
        if merged_chunks and len(chunk) < MIN_CHUNK_CHARS:
            candidate = f"{merged_chunks[-1]}\n\n{chunk}"
            if len(candidate) <= MAX_CHUNK_CHARS:
                merged_chunks[-1] = candidate
                continue
        merged_chunks.append(chunk)
    return merged_chunks


class RagIndex:
    def __init__(self, chunks: list[str], vectors: list[list[float]]):
        self.chunks = chunks
        self.vectors = vectors

    def retrieve(self, query_vec, k: int = TOP_K) -> list[str]:
        scored = [(_cosine(query_vec, vec), i) for i, vec in enumerate(self.vectors)]
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:k]]


def _build_index() -> RagIndex:
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Нет базы знаний: {KB_PATH}")
    text = KB_PATH.read_text(encoding="utf-8")
    embeddings = _embeddings_client()
    chunks = _semantic_chunks(_split_units(text), embeddings)
    if not chunks:
        raise ValueError("knowledge_base.txt пуст, чанки не получены")
    vectors = embeddings.embed_documents(chunks)
    index = RagIndex(chunks, vectors)
    CACHE_PATH.write_bytes(
        pickle.dumps(
            {
                "mtime": KB_PATH.stat().st_mtime,
                "model": EMBED_MODEL,
                "chunks": chunks,
                "vectors": vectors,
            }
        )
    )
    return index


def load_index() -> RagIndex:
    global _index
    if _index is not None:
        return _index
    if CACHE_PATH.exists() and KB_PATH.exists():
        data = pickle.loads(CACHE_PATH.read_bytes())
        same_kb = data.get("mtime") == KB_PATH.stat().st_mtime
        same_model = data.get("model") == EMBED_MODEL
        if same_kb and same_model:
            _index = RagIndex(data["chunks"], data["vectors"])
            return _index
    _index = _build_index()
    return _index


def get_rag_context(query: str, k: int = TOP_K) -> str:
    index = load_index()
    query_vec = _embeddings_client().embed_query(query)
    hits = index.retrieve(query_vec, k=k)
    return "\n\n".join(f"[Фрагмент {i}]\n{chunk}" for i, chunk in enumerate(hits, 1))


if __name__ == "__main__":
    # консоль Windows в cp1251 не печатает эмодзи из путеводителя
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    probe = _embeddings_client().embed_query("тест")
    print(f"эмбеддинги: {EMBED_MODEL} @ {EMBED_BASE}, размерность {len(probe)}")
    index = load_index()
    print(f"чанков: {len(index.chunks)}")
    for i, chunk in enumerate(index.chunks, 1):
        preview = chunk.replace("\n", " ")[:160]
        print(f"\n--- {i} ({len(chunk)} символов) ---\n{preview}")
