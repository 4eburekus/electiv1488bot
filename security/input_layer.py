"""Слой 1. Вход: что бот принимает от пользователя.

Перенесено из secure_ai_agents-main/student/input_layer.py без изменений:
Telegram позволяет прислать до 4096 символов и любые невидимые символы, поэтому
нормализация Юникода и фильтр длины стоят ДО обращения к модели.

Два отличия от учебника: значения по умолчанию читаются из .env (чтобы лимиты
настраивались без правки кода) и RateLimiter завязан на chat_id, а не на вымышленного
пользователя.
"""
import os
import unicodedata

# Нулевая ширина (U+200B…U+200F), WORD JOINER, BOM, символы-теги U+E0000…U+E007F и
# управляющие символы — всё это категории "C": Cc/Cf/Co/Cs/Cn. Резать по категории
# надёжнее, чем перечислять кодовые точки: невидимых символов в Юникоде больше сотни.
_INVISIBLE = {"Cc", "Cf", "Co", "Cs", "Cn"}
_KEPT = {"\n", "\t"}

# Запрос длиннее этого не уходит к модели: в нём уже помещается полезная нагрузка
# инъекции, а обрезка молча меняет смысл вопроса.
DEFAULT_MAX_LEN = int(os.getenv("SEC_INPUT_MAX_LEN", "2000"))
# Сколько запросов один chat_id может сделать за окно.
DEFAULT_RATE_LIMIT = int(os.getenv("SEC_RATE_LIMIT", "20"))
DEFAULT_RATE_WINDOW_S = float(os.getenv("SEC_RATE_WINDOW_S", "60"))


class InputRejected(Exception):
    """Запрос не принят: слишком длинный, пустой или состоит из мусора."""


def _strip_invisible(text: str) -> str:
    return "".join(ch for ch in text if ch in _KEPT or unicodedata.category(ch) not in _INVISIBLE)


def sanitize_input(text: str, max_len: int = DEFAULT_MAX_LEN) -> str:
    # Два прохода: сам NFKC может раскрыть невидимый символ (например полноширинный
    # пробел U+3000 или лигатуру), поэтому чистим уже после нормализации тоже.
    cleaned = _strip_invisible(unicodedata.normalize("NFKC", _strip_invisible(text)))

    if not cleaned.strip():
        raise InputRejected("пустой запрос")
    if len(cleaned) > max_len:
        # Молча обрезать нельзя: обрезанный запрос может поменять смысл.
        raise InputRejected(f"запрос длиннее {max_len} символов")
    return cleaned


class RateLimiter:
    def __init__(self, limit: int = DEFAULT_RATE_LIMIT, window_s: float = DEFAULT_RATE_WINDOW_S):
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, list[float]] = {}   # chat_id -> времена принятых запросов

    def allow(self, user_id: str, now: float) -> bool:
        # Скользящее окно: оставляем только запросы моложе window_s секунд.
        fresh = [t for t in self._hits.get(user_id, ()) if now - t < self.window_s]
        if len(fresh) >= self.limit:
            self._hits[user_id] = fresh
            return False
        fresh.append(now)
        self._hits[user_id] = fresh
        return True
