""" Черновой вариант из ноутбука

Индексация:
1. Загрузка всех PDF из директории с помощью PyPDFLoader + статистика
2. Чанкование с помощью RecursiveCharacterTextSplitter + статистика
3. Эмбеддинги - Превращение текста чанков в векторы
4. Создание chromadb индекса, Сохранение и загрузка chromadb индекса в папку базы знаний
"""
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
)
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from langchain_chroma import Chroma
import chromadb
import shutil
from rag import embeddings_rag_indexing


pdf_folder = Path("./knowledge_sources")
pdf_files = sorted(pdf_folder.glob("*.pdf"))
print(f"📁 Найдено PDF файлов: {len(pdf_files)}")


# Загрузка
dir_loader = DirectoryLoader(
    path=pdf_folder,
    glob="*.pdf",
    loader_cls=PyPDFLoader,
    use_multithreading=True,
    show_progress=True
)
all_docs = dir_loader.load()
print(f"Загружено документов: {len(all_docs)}")

# Статистика загрузки
# import re
# import time
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
# total_time = sum(r["pypdf: сек"] for r in loader_rows)
# avg_single_letters = sum(r["pypdf: слов из 1 буквы"] for r in loader_rows) / len(loader_rows)
# print(f"   • Общее время загрузки всей базы: {total_time:.1f} сек.")
# print(f"   • Средняя доля слов из одной буквы по всем файлам: {avg_single_letters:.1%}")
# print("   • Доля слов из одной буквы находится в пределах нормы (2-3%), что свидетельствует об отсутствии рассыпания слов.")


# Чанкование с помощью RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = text_splitter.split_documents(all_docs)
# Подписи источников: по ним модель (и мы) смогут сослаться на конкретную заметку
document_metadatas = [{"source": f"заметка {i}"} for i in range(1, len(chunks) + 1)]

# Статистика
print(f"📊 Статистика чанков:")
print(f"  Всего чанков: {len(chunks)}")
print(f"  Средняя длина: {np.mean([len(c.page_content) for c in chunks]):.1f} символов")
print(f"  Мин. длина: {np.min([len(c.page_content) for c in chunks])} символов")
print(f"  Макс. длина: {np.max([len(c.page_content) for c in chunks])} символов")


# Эмбеддинги
embeddings_model = embeddings_rag_indexing()
test_embedding = embeddings_model.embed_query("проверка размерности")
print(f"📏 Размерность эмбеддингов модели: {len(test_embedding)}")


# Создание Chroma хранилища с использованием абсолютного пути
chroma_db_path = os.path.abspath("chroma_db")
# создаём папку на диске
if os.path.exists(chroma_db_path):
    shutil.rmtree(chroma_db_path)
print("📚 Создание Chroma хранилища...")
chroma_store = Chroma.from_texts(
    texts=[chunk.page_content for chunk in chunks],
    embedding=embeddings_model,
    metadatas=document_metadatas,
    persist_directory=chroma_db_path,
    collection_name="rag_educational_collection"
)
print(f"✅ Chroma хранилище создано!")
print(f"📊 Количество документов: {chroma_store._collection.count()}")



# Загрузка chromadb в инструмент

# chromadb >= 0.4 сохраняет данные автоматически при указании persist_directory
# print("💾 Chroma автоматически сохраняет данные (chromadb >= 0.4).")
# print("✅ Данные сохранены!")
# print(f"\n📥 Загрузка хранилища...")
# loaded_chroma = Chroma(
#     collection_name="rag_educational_collection",
#     persist_directory=chroma_db_path,
#     embedding_function=embeddings_model
# )
# print(f"✅ Хранилище загружено!")
# print(f"📊 Количество документов: {loaded_chroma._collection.count()}")



# Проверка индексации
# if __name__ == "__main__":









