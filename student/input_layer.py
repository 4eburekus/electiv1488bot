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
import unicodedata  # noqa: F401  (пригодится в решении)


class InputRejected(Exception):
    """Запрос не принят: слишком длинный, пустой или состоит из мусора."""


def sanitize_input(text: str, max_len: int = 2000) -> str:
    # TODO: реализуйте очистку и проверки из описания выше.
    return text


class RateLimiter:
    def __init__(self, limit: int = 5, window_s: float = 60):
        self.limit = limit
        self.window_s = window_s

    def allow(self, user_id: str, now: float) -> bool:
        # TODO: храните время прошлых запросов каждого пользователя и считайте только свежие.
        return True
