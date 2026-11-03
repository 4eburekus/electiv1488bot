"""Индексация:
1. Загрузка всех PDF из директории: для каждого файла текст извлекается двумя
   загрузчиками (pypdf и pdfplumber), выбирается тот, где меньше искажений + статистика
2. Нормализация текста и чанкование RecursiveCharacterTextSplitter + статистика
3. Эмбеддинги - Превращение текста чанков в векторы
4. Создание нового индекса во ВРЕМЕННУЮ КОЛЛЕКЦИЮ рабочей базы (старая цела)
5. Проверка нового индекса: count + тестовый similarity_search
6. Свап коллекций (delete старой + rename новой) и финальное подтверждение

Порядок 4 -> 5 -> 6 выбран так, чтобы ошибка сети/лимитов/ключей на этапе эмбеддингов
никогда не оставляла нас без базы знаний. Свап сделан на уровне коллекций, а не
перемещением папки: на Windows chromadb держит файл базы открытым, и os.rename
директории падает с PermissionError даже после client.close().

Шаг 1 с двумя загрузчиками нужен потому, что оба ломают текст по-разному и лечится
это только посимвольно: pypdf переставляет токены в листингах кода
(`name="manager_agent",` -> `name= ,`), а pdfplumber на двухколоночных статьях склеивает
соседние слова (`Lin BY, Fu Y` -> `LinBY,FuY`) и теряет 5-7% токенов. Поэтому для каждого
PDF считается штраф за искажения и берётся тот вариант, где он меньше.
"""
import hashlib
import re
import time
from pathlib import Path

import chromadb
import numpy as np
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import EMB_LLM_MODEL_NAME, embeddings_rag_indexing

# Всё привязано к файлу, а не к текущей директории запуска: скрипт можно запускать откуда угодно
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "knowledge_sources"
DB_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "rag_educational_collection"
STAGING_COLLECTION_NAME = f"{COLLECTION_NAME}__staging"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
VERIFY_QUERY = "как агент должен вызывать инструменты"
# pdfplumber медленнее pypdf примерно в 2 раза, поэтому берём его только при заметной победе
LOADER_MARGIN = 0.5

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё'`]+")
GLUED_RE = re.compile(r"[a-zа-яё][A-ZА-ЯЁ]")       # LinBY,FuY / arXivpreprint
BROKEN_ARG_RE = re.compile(r"=\s*[,\)]")           # name= ,  x= )
EMPTY_CALL_RE = re.compile(r"\(\s{0,8}\)")         # instructions=( ... ) пустой вызов


def normalize_text(text: str) -> str:
    """Прибираем мусор PDF-вёрстки, не трогая содержимое.

    \xa0 (неразрывный пробел) и \u2028/\u2029 (LINE SEPARATOR) — артефакты вёрстки:
    для эмбеддинга это почти не мешает, но они не совпадают с обычным пробелом при
    поиске подстрокой и раздувают чанки пустыми строками.
    """
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("\xa0", " ").replace("\t", "    ")
    lines = [line.rstrip() for line in text.split("\n")]

    out: list[str] = []
    prev_blank = True
    for line in lines:
        if not line.strip():
            if not prev_blank:  # схлопнуть несколько пустых строк в одну, ведущие выкинуть
                out.append("")
                prev_blank = True
            continue
        out.append(line)
        prev_blank = False
    return "\n".join(out).strip("\n")


def damage_score(text: str) -> float:
    """Штраф за искажения текста на 1000 символов. Меньше — лучше.

    Не метрика «красоты», а способ отличить перестановку токенов в коде от склейки
    слов в двухколоночной статье — оба дефекта лечатся только сменой загрузчика.
    """
    if not text:
        return float("inf")
    words = WORD_RE.findall(text)
    glued = len(GLUED_RE.findall(text))
    long_tokens = sum(1 for w in words if len(w) > 22)          # scientsstacksweep
    single_letter_share = sum(1 for w in words if len(w) == 1) / max(len(words), 1)
    broken_args = len(BROKEN_ARG_RE.findall(text))
    empty_calls = len(EMPTY_CALL_RE.findall(text))
    penalty = glued + 3 * long_tokens + 8 * broken_args + 8 * empty_calls
    # «разорванность»: доля строк из одного слова. У нормального текста это заголовки
    # (5-7%), но у плохого извлечению каждый символ уезжает на свою строку
    # (ai_agent_bible.pdf через pypdf даёт 100%), и тогда чанк из 800 символов
    # содержит около сотни слов вместо нескольких абзацев.
    lines = [line for line in text.split("\n") if line.strip()]
    single_word_share = sum(1 for line in lines if len(line.split()) == 1) / max(len(lines), 1)
    return penalty * 1000 / max(len(text), 1) + 25 * single_letter_share + 10 * single_word_share


def _extract(path: Path, loader_cls) -> list[Document]:
    """Загрузить один PDF, нормализовать страницы и выбросить пустые."""
    docs = []
    for doc in loader_cls(str(path)).load():
        doc.page_content = normalize_text(doc.page_content)
        if doc.page_content.strip():
            docs.append(doc)
    return docs


def load_documents() -> list[Document]:
    """Шаг 1: загрузка PDF постранично с выбором загрузчика по качеству текста."""
    if not KB_DIR.is_dir():
        raise FileNotFoundError(f"Нет папки с источниками: {KB_DIR}")

    pdf_files = sorted(KB_DIR.glob("*.pdf"))
    print(f"📁 Найдено PDF файлов: {len(pdf_files)}")
    if not pdf_files:
        raise FileNotFoundError(f"В {KB_DIR} нет ни одного .pdf — строить индекс не из чего")

    t0 = time.time()
    all_docs: list[Document] = []
    rows = []
    for path in pdf_files:
        pypdf_docs = _extract(path, PyPDFLoader)
        plumber_docs = _extract(path, PDFPlumberLoader)
        s_pypdf = damage_score("\n".join(d.page_content for d in pypdf_docs))
        s_plumber = damage_score("\n".join(d.page_content for d in plumber_docs))

        if s_plumber <= s_pypdf - LOADER_MARGIN:
            chosen, label, score = plumber_docs, "pdfplumber", s_plumber
        else:
            chosen, label, score = pypdf_docs, "pypdf", s_pypdf
        all_docs.extend(chosen)
        rows.append((path.name, label, s_pypdf, s_plumber, len(chosen)))

    print(f"Извлечено страниц: {len(all_docs)} за {time.time() - t0:.1f} сек.")
    print("🧪 Выбор загрузчика по штрафу за искажения (меньше = чище):")
    for name, label, s_py, s_pl, pages in rows:
        flag = "⚠️ " if label == "pdfplumber" else "   "
        print(f"  {flag}{name[:46]:48} pypdf={s_py:6.2f} plumber={s_pl:6.2f} -> {label:11} стр.{pages}")
    return all_docs


# Статистика загрузки
# import pandas as pd
# def _single_letter_share(text):
#     """Доля «слов» из одной буквы. У нормального текста мала (a, I, в, с, и), у рассыпанного — велика."""
#     words = text.split()
#     return sum(len(w) == 1 and w.isalpha() for w in words) / max(len(words), 1)
# def _loader_report(path):
#     row = {"файл": path.name}
#     t0 = time.time()
#     full_text = "\n".join(d.page_content for d in PyPDFLoader(str(path)).load())
#     row["pypdf: символов без пробелов"] = len(re.sub(r"\s+", "", full_text))
#     row["pypdf: доля слов из 1 буквы"] = _single_letter_share(full_text)
#     row["pypdf: сек"] = time.time() - t0
#     return row
# print("Доля слов из одной буквы в пределах нормы: 2-3%")
# loader_rows = [_loader_report(p) for p in pdf_files]
# loader_df = pd.DataFrame(loader_rows)
# print(loader_df)


def split_documents(all_docs: list[Document]) -> list[Document]:
    """Шаг 2: чанкование. Метаданные источника (файл, страница) сохраняются от загрузчика."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = [c for c in text_splitter.split_documents(all_docs) if c.page_content.strip()]
    if not chunks:
        raise RuntimeError("После чанкования не осталось ни одного непустого чанка")

    # Стабильный id: хэш содержимого, а не позиция в списке.
    # Добавление/удаление PDF не сдвигает идентификаторы остальных чанков,
    # и повторная индексация не создаёт дублей.
    unique: dict[str, Document] = {}
    for chunk in chunks:
        source = Path(str(chunk.metadata.get("source", ""))).name
        page = int(chunk.metadata.get("page", 0)) + 1  # для человека страницы нумеруются с 1
        digest = hashlib.sha1(f"{source}\n{page}\n{chunk.page_content}".encode("utf-8")).hexdigest()[:16]
        if digest in unique:
            continue
        # Оставляем только свои поля: у pypdf и pdfplumber разные наборы служебных
        # метаданных (Producer/ModDate/page_label...), в базе они лишь раздувают запись.
        chunk.metadata = {"source": source, "page": page, "id": digest}
        unique[digest] = chunk

    if len(unique) != len(chunks):
        print(f"♻️  Убрано дубликатов чанков: {len(chunks) - len(unique)}")
    return list(unique.values())


def print_chunk_stats(chunks: list[Document]) -> None:
    """Статистические результаты чанкования"""
    lengths = [len(c.page_content) for c in chunks]
    sources = {c.metadata["source"] for c in chunks}
    lines = sum(len(c.page_content.split("\n")) for c in chunks)
    blank = sum(1 for c in chunks for ln in c.page_content.split("\n") if ln and ln.isspace())
    print("📊 Статистика чанков:")
    print(f"  Всего чанков: {len(chunks)}")
    print(f"  Уникальных источников: {len(sources)}")
    print(f"  Средняя длина: {np.mean(lengths):.1f} символов")
    print(f"  Мин. длина: {np.min(lengths)} символов")
    print(f"  Макс. длина: {np.max(lengths)} символов")
    print(f"  Строк только из пробелов: {blank} из {lines}")


def check_embeddings(embeddings_model) -> int:
    """Шаг 3: эмбеддинги — проверяем размерность ДО записи в базу."""
    test_embedding = embeddings_model.embed_query("проверка размерности")
    dim = len(test_embedding)
    if dim == 0:
        raise RuntimeError("Модель вернула пустой вектор")
    print(f"📏 Размерность эмбеддингов модели: {dim}")
    return dim


def build_index(chunks: list[Document], embeddings_model) -> int:
    """Шаг 4: запись во временную коллекцию. Рабочая коллекция на этом шаге не трогается.

    Возвращает количество записанных чанков."""
    if not EMB_LLM_MODEL_NAME:
        raise RuntimeError("Не задан EMB_LLM_MODEL_NAME — не что записать в метаданные коллекции")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if any(c.name == STAGING_COLLECTION_NAME for c in client.list_collections()):
        client.delete_collection(STAGING_COLLECTION_NAME)

    print("📚 Создание Chroma хранилища (временная коллекция)...")
    t0 = time.time()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        ids=[c.metadata["id"] for c in chunks],
        persist_directory=str(DB_DIR),
        collection_name=STAGING_COLLECTION_NAME,
        # chromadb НЕ хранит, какая модель дала векторы — фиксируем сами,
        # иначе rag.py сможет молча искать другой моделью по старой базе
        collection_metadata={"embedding_model": EMB_LLM_MODEL_NAME, "hnsw:space": "cosine"},
    )
    print(f"✅ Временная коллекция создана за {time.time() - t0:.1f} сек.")
    del store  # дальше читаем коллекцию с диска отдельным клиентом
    return client.get_collection(STAGING_COLLECTION_NAME).count()


def verify_index(
    collection_name: str,
    expect_model: str,
    expect_count: int,
    query: str,
    embeddings_model,
) -> None:
    """Шаг 5: независимая проверка — читаем коллекцию с диска так же, как это сделает rag.py."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(collection_name)

    count = collection.count()
    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != expect_model:
        raise RuntimeError(
            f"Индекс построен моделью {stored_model!r}, а сейчас используется {expect_model!r}: "
            "векторы несопоставимы, нужна полная пересборка"
        )
    if count == 0:
        raise RuntimeError("В индексе ноль документов")
    if count != expect_count:
        raise RuntimeError(f"В индексе {count} чанков, ожидалось {expect_count}")

    hits = collection.query(
        query_embeddings=[embeddings_model.embed_query(query)],
        n_results=1,
        include=["documents", "metadatas", "distances"],
    )
    meta_hit = hits["metadatas"][0][0]
    print("🔎 Тестовый запрос по открытой с диска базе:")
    print(f"  коллекция={collection_name} count={count} модель={stored_model} "
          f"distance={hits['distances'][0][0]:.4f}")
    print(f"  top-hit id={hits['ids'][0][0]} {meta_hit.get('source')} стр.{meta_hit.get('page')}")
    print(f"  фрагмент: {hits['documents'][0][0][:120]!r}")


def swap_index() -> None:
    """Шаг 6: временная коллекция становится рабочей. Старая к этому моменту ещё не удалена."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if any(c.name == COLLECTION_NAME for c in client.list_collections()):
        client.delete_collection(COLLECTION_NAME)
    staging = client.get_collection(STAGING_COLLECTION_NAME)
    staging.modify(name=COLLECTION_NAME)
    print(f"💾 Коллекция {COLLECTION_NAME} сохранена в {DB_DIR}")


def main() -> None:
    all_docs = load_documents()                    # 1
    chunks = split_documents(all_docs)             # 2
    print_chunk_stats(chunks)

    embeddings_model = embeddings_rag_indexing()   # 3
    check_embeddings(embeddings_model)

    count = build_index(chunks, embeddings_model)  # 4
    print(f"📊 Записано чанков: {count} из {len(chunks)}")

    verify_index(STAGING_COLLECTION_NAME, EMB_LLM_MODEL_NAME, len(chunks), VERIFY_QUERY, embeddings_model)  # 5
    swap_index()                                                                                            # 6
    verify_index(COLLECTION_NAME, EMB_LLM_MODEL_NAME, len(chunks), VERIFY_QUERY, embeddings_model)           # 6b


if __name__ == "__main__":
    main()
