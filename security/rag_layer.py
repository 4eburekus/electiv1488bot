"""Слой 3. База знаний: приём документов в индекс.

clean_document перенесён из secure_ai_agents-main/student/rag_layer.py пологично:
HTML-комментарии, элементы со скрытым стилем (display:none, visibility:hidden, font:0,
белый текст) и символы нулевой ширины. Видимое содержимое сохраняется.

Отличие одно и оно принципиальное: в учебнике источник чанка начинается с "kb/", потому что
индекс собран из локальной папки с префиксом. В electiv1488bot в метаданных лежит голое имя
файла (tool_rag/rag_indexing.py берёт Path(source).name), поэтому правило переписано под наш
формат: имя PDF без пути, без «..» и без управляющих символов.

retrieve(query, user, ...) из учебника не перенесён: прав доступа по группам у бота нет,
как нет и второго читателя — один источник знаний и все вопросы приходят из чата.
"""
import re

# Скрытое содержимое: то, что модель прочитает, а человек нет.
_COMMENT = re.compile(r"<!--.*?-->", re.S)
# U+200B…U+200F (zero-width space/joiners/marks), U+2060 WORD JOINER, U+FEFF BOM,
# U+180E MONGOLIAN VOWEL SEPARATOR. Записаны escape-последовательностями, а не самими
# символами: невидимый символ в исходнике легко потерять при копировании и не заметить.
_ZERO_WIDTH = re.compile("[\\u200b-\\u200f\\u2060\\ufeff\\u180e]")
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

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def is_trusted_source(source) -> bool:
    """Принимать в выдачу только имя нашего PDF, и ничего похожего на обход пути."""
    if not isinstance(source, str) or not source.strip():
        return False
    name = source.strip()
    # Разделители пути и «..»: источник не должен указывать никуда вне папки базы знаний.
    if "/" in name or "\\" in name or ".." in name:
        return False
    if _CONTROL_CHARS.search(name):
        return False
    # Формат метаданных rag_indexing.py — имя файла с расширением. Всё остальное (URL,
    # «stdin», «web: …») появилось в источнике не из нашей папки, значит его подмешали.
    return name.lower().endswith(".pdf")


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
