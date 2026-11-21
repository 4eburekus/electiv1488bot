"""Задание 1. Вход: что бот принимает от пользователя (глава III лекции).

Сейчас бот пропускает любой текст любой длины от любого числа запросов. Исправьте:

  1. sanitize_input(text, max_len)
       - вырезать невидимые символы: нулевой ширины (U+200B…U+200F, U+2060, U+FEFF) и
         «символы-теги» U+E0000…U+E007F, а также управляющие символы (кроме \\n и \\t);
       - привести текст к нормальной форме Юникода (unicodedata.normalize("NFKC", ...)):
         так «ｉｇｎｏｒｅ» из полноширинных букв превращается в обычное «ignore»;
       - если после очистки текст пустой или длиннее max_len — выбросить InputRejected.
         Не обрезайте молча: обрезанный запрос может поменять смысл.
  2. RateLimiter — не больше `limit` запросов от одного пользователя за `window_s` секунд.
       Окно скользящее: запросы старше window_s секунд не считаются. Пользователи независимы.

Проверка: pytest tests/test_01_input.py
"""
import unicodedata

# Нулевая ширина (U+200B…U+200F), WORD JOINER, BOM, символы-теги U+E0000…U+E007F и
# управляющие символы — всё это категории "C": Cc/Cf/Co/Cs/Cn. Резать по категории
# надёжнее, чем перечислять кодовые точки: невидимых символов в Юникоде больше сотни.
_INVISIBLE = {"Cc", "Cf", "Co", "Cs", "Cn"}
_KEPT = {"\n", "\t"}


class InputRejected(Exception):
    """Запрос не принят: слишком длинный, пустой или состоит из мусора."""


def _strip_invisible(text: str) -> str:
    return "".join(ch for ch in text if ch in _KEPT or unicodedata.category(ch) not in _INVISIBLE)


def sanitize_input(text: str, max_len: int = 2000) -> str:
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
    def __init__(self, limit: int = 5, window_s: float = 60):
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, list[float]] = {}   # user_id -> времена принятых запросов

    def allow(self, user_id: str, now: float) -> bool:
        # Скользящее окно: оставляем только запросы моложе window_s секунд.
        fresh = [t for t in self._hits.get(user_id, ()) if now - t < self.window_s]
        if len(fresh) >= self.limit:
            self._hits[user_id] = fresh
            return False
        fresh.append(now)
        self._hits[user_id] = fresh
        return True
