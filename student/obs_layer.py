"""Задание 7. Наблюдаемость: журнал без содержимого, тревога, выключатель (глава VII).

  1. audit_event(user_id, event, text) -> dict — запись журнала. Сейчас в неё кладётся весь
     текст сообщения: журнал превращается в склад чужих писем и паролей. Оставьте поля
     user_id, event, chars (длина текста) и sha256 (первые 12 символов хеша: по нему можно
     найти повтор одного и того же запроса, не храня сам запрос). Поля "text" быть не должно.
  2. AlertDetector — тревога по числу запросов в минуту. observe(count) вызывается раз в минуту
     и возвращает True, если число запросов резко выше обычного:
       - обычное = среднее по последним `window` минутам без тревоги (минуты с тревогой
         в базу сравнения не попадают, иначе всплеск «приучит» детектор к себе);
       - пока накоплено меньше 5 минут истории, тревоги нет;
       - тревога, если count > factor * обычное.
     Подберите window и factor так, чтобы на шумном графике из tests/test_07_observability.py
     не было ложных тревог и всплеск ×4 замечался не позже чем через 2 минуты.
  3. kill_switch_active(path) -> bool — выключатель снаружи модели: бот отключён, пока на диске
     существует файл `path`. Агент не может «передумать» и обойти проверку, ведь она в коде.

Проверка: pytest tests/test_07_observability.py
"""
import hashlib  # noqa: F401
import os  # noqa: F401


def audit_event(user_id: str, event: str, text: str) -> dict:
    # TODO: уберите text, добавьте chars и sha256.
    return {"user_id": user_id, "event": event, "text": text}


class AlertDetector:
    def __init__(self, window: int = 10, factor: float = 2.5):
        self.window = window
        self.factor = factor
        self.history = []

    def observe(self, count: float) -> bool:
        # TODO: сравните count с обычным уровнем.
        return False


def kill_switch_active(path: str) -> bool:
    # TODO: проверьте существование файла.
    return False
