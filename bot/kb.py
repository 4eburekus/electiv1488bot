"""База знаний и простейший поиск.

Поиск здесь намеренно примитивный (совпадение «основ» слов): лаборатория про безопасность,
а не про качество поиска. Эмбеддинги и реранкер вы уже разбирали в common_ai_agents.
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Читатели бота. groups — к каким закрытым разделам у человека есть доступ.
USERS = {
    "anna": {"id": "anna", "name": "Анна (отдел кадров)", "groups": ["hr", "all"]},
    "boris": {"id": "boris", "name": "Борис (разработчик)", "groups": ["dev", "all"]},
    "guest": {"id": "guest", "name": "Гость", "groups": ["all"]},
}


def load_raw_chunks() -> list:
    """Чанки как они лежат в файле: {id, text, group, source}."""
    return json.loads((DATA / "kb.json").read_text(encoding="utf-8"))


_WORD = re.compile(r"[a-zа-яё0-9]+", re.I)


def stems(text: str) -> set:
    """Грубая «основа» слова: первые пять букв. Слова короче четырёх букв отбрасываем."""
    return {w[:5] for w in _WORD.findall(text.lower()) if len(w) >= 4}


def score(query: str, text: str) -> int:
    """Сколько основ слов из запроса встретилось в тексте чанка."""
    return len(stems(query) & stems(text))
