"""Задание 3. База знаний: права читателя и приём документов (глава IV).

Сейчас поиск не знает, кто спрашивает: гость получает зарплатные ведомости. Документы
принимаются в индекс без проверки: из них можно подложить скрытую инструкцию. Исправьте:

  1. retrieve(query, user, chunks, k, score) — вернуть k самых подходящих чанков, но
     ТОЛЬКО из доступных читателю: у чанка есть поле "group", у читателя — список user["groups"].
     Фильтр применяется ДО отбора лучших k. Если сначала взять топ-k, а потом отфильтровать,
     читатель может остаться с пустой выдачей, и это ломает стенд ACL из лекции.
     score(query, text) — готовая функция оценки релевантности. Сортируйте по убыванию оценки.
  2. is_trusted_source(source) — доверять только источникам, начинающимся с "kb/" и без «..».
  3. clean_document(text) — очистить текст при приёме в индекс: убрать HTML-комментарии
     <!-- ... -->, элементы со стилем display:none или белым цветом текста
     (color:#fff / color:white), символы нулевой ширины. Видимый текст сохранить.

Проверка: pytest tests/test_03_rag.py
"""
import re

# Группа чанка без явной группы считается открытой: в индексе лаборатория хранит только
# свои документы, а «свой» раздел у читателя перечислен в user["groups"].
_DEFAULT_GROUP = "all"

# Скрытое содержимое: то, что модель прочитает, а человек нет.
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿᠎]")
# Стиль, делающий текст невидимым: спрятан совсем или сольётся с фоном.
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none"
    r"|visibility\s*:\s*hidden"
    r"|font\s*:\s*0(?:px)?"
    r"|color\s*:\s*(?:#(?:fff|ffffff)\b|white\b|rgb\(\s*255\s*,\s*255\s*,\s*255\s*\))",
    re.I,
)
# Парный элемент: <tag ... style="..."> ... </tag>
_ELEMENT = re.compile(r"<([a-zA-Z][\w:-]*)\b([^>]*)>(.*?)</\1\s*>", re.S | re.I)
_STYLE_ATTR = re.compile(r"""style\s*=\s*(["'])(.*?)\1""", re.S | re.I)


def retrieve(query: str, user: dict, chunks: list, k: int, score) -> list:
    groups = set((user or {}).get("groups") or ())
    # Сначала права, потом релевантность. Наоборот — и топ-k окажется занят чужими чанками,
    # а читатель с узкими правами получит пустую выдачу.
    allowed = [c for c in chunks if c.get("group", _DEFAULT_GROUP) in groups]
    # Оценка считается один раз на чанк; по идентичной оценке порядок фиксирован по id,
    # чтобы выдача не зависела от порядка строк в файле.
    ranked = sorted(
        ((-score(query, c["text"]), c.get("id", ""), c) for c in allowed),
        key=lambda item: item[:2],
    )
    return [c for _, _, c in ranked[:max(k, 0)]]


def is_trusted_source(source: str) -> bool:
    # Только наш индекс и ничего похожего на обход пути: "kb/../upload/x" начинается с "kb/",
    # но указывает наружу. Загруженное пользователем (upload/...) инструкций не даёт.
    return isinstance(source, str) and source.startswith("kb/") and ".." not in source


def _strip_hidden(text: str) -> str:
    """Убирает элементы со скрытым стилем целиком, вместе с их невидимым содержимым."""
    out, pos = [], 0
    for m in _ELEMENT.finditer(text):
        style = _STYLE_ATTR.search(m.group(2))
        out.append(text[pos:m.start()])
        if not (style and _HIDDEN_STYLE.search(style.group(2))):
            # Элемент открытый: содержимое сохраняем, но и в нём режем скрытые части.
            out.append(_strip_hidden(m.group(3)))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def clean_document(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _COMMENT.sub(" ", text)      # инструкция, спрятанная в комментарии
    cleaned = _strip_hidden(cleaned)        # белый на белом и display:none
    cleaned = _ZERO_WIDTH.sub("", cleaned)  # символы нулевой ширины
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()
