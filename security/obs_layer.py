"""Слой 7. Наблюдаемость: журнал без содержимого, тревога, выключатель.

audit_event, AlertDetector и kill_switch_active — из
secure_ai_agents-main/student/obs_layer.py дословно.

Добавлен log_event(): в electiv1488bot вместо print-строк в консоль журнала нет вообще,
поэтому записи складываются в JSON Lines рядом с ботом. Содержимое запроса туда не попадает
ни в каком виде — только длина и отпечаток.
"""
import hashlib
import json
import os
import time

# Длина хеша в записи: 12 шестнадцатеричных разрядов — 48 бит. Повтор запроса находится по
# такому отпечатку однозначно, а восстановить по нему текст нельзя даже теоретически.
HASH_PREFIX = 12

# Хвост графика, который детектор помнит как «обычно». 10 минут достаточно, чтобы шум внутри
# одной минуты (20 ± 6) не сошёл за всплеск, и мало, чтобы не протухнуть после смены режима.
DEFAULT_WINDOW = 10
# Порог «в 2.5 раза выше обычного». Фон даёт 14…26 при среднем ~20, всплеск ×4 — 56…104:
# между ними остаётся зазор, в который порог попадает с запасом с обеих сторон.
DEFAULT_FACTOR = 2.5
# Пока истории меньше этого, сравнивать не с чем — тревоги не будет (иначе первая минута
# графика с любым ненулевым трафиком выглядит атакой).
MIN_HISTORY = 5

AUDIT_LOG_PATH = os.getenv("SEC_AUDIT_LOG", "security_audit.jsonl")
KILL_SWITCH_FILE = os.getenv("SEC_KILL_SWITCH_FILE", "STOP_BOT")


def audit_event(user_id: str, event: str, text: str) -> dict:
    body = text if isinstance(text, str) else str(text or "")
    # Содержимое в журнал не попадает: он читается людьми и живёт дольше запроса. Остаётся
    # метрика (chars) и отпечаток (sha256) — по нему видно, что тот же запрос приходил
    # повторно, но что в нём было — не восстановить.
    return {
        "ts": round(time.time(), 3),
        "user_id": user_id,
        "event": event,
        "chars": len(body),
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()[:HASH_PREFIX],
    }


def log_event(user_id: str, event: str, text: str = "") -> dict:
    """Строка журнала в файл. Путь пустой — значит журнал отключён (для офлайн-проверок)."""
    record = audit_event(user_id, event, text)
    path = os.getenv("SEC_AUDIT_LOG", AUDIT_LOG_PATH)
    if not path:
        return record
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Журнал не должен ронять ответ пользователю: диск, права и ротация — не его работа.
        pass
    return record


class AlertDetector:
    """Скользящее сравнение текущего трафика с «обычным» уровнем прошлых минут."""

    def __init__(self, window: int = DEFAULT_WINDOW, factor: float = DEFAULT_FACTOR):
        self.window = window
        self.factor = factor
        self.history = []   # только спокойные минуты: всплеск не должен учить детектор себе

    def observe(self, count: float) -> bool:
        value = float(count)
        # Тревожная минута в базу не идёт. Иначе 16 минут подряд по 80 запросов подняли бы
        # «обычный уровень» до 80, и детектор замолчал бы ровно посреди атаки.
        if len(self.history) >= MIN_HISTORY:
            usual = sum(self.history[-self.window:]) / len(self.history[-self.window:])
            if value > self.factor * usual:
                return True
        self.history.append(value)
        if self.window > 0 and len(self.history) > self.window:
            # Храним не больше window минут: старое «обычное» со временем протухает.
            del self.history[:-self.window]
        return False


class TrafficMonitor:
    """Скользящий счётчик запросов по минутам поверх AlertDetector.

    AlertDetector из учебника принимает уже готовое число за минуту; здесь — то, чем
    бот реально располагает: поток сообщений. Минута закрылась — сравниваем её с
    обычным уровнем и возвращаем True, если трафик подскочил.

    Никуда не пишет и ничего не блокирует: тревога — сигнал в журнал, а не отказ
    пользователю (для отказа есть RateLimiter).
    """

    def __init__(self, detector: "AlertDetector | None" = None, bucket_s: float = 60.0):
        self.detector = detector or AlertDetector()
        self.bucket_s = bucket_s
        self._bucket = None
        self.count = 0          # запросов в текущей минуте — для текста тревоги

    def observe(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        bucket = int(now // self.bucket_s)
        if self._bucket is None:
            self._bucket = bucket
        spike = False
        if bucket != self._bucket:
            # Минута закрылась: сравниваем её с обычным уровнем и начинаем новую.
            spike = self.detector.observe(self.count)
            self._bucket = bucket
            self.count = 0
        self.count += 1
        return spike


def kill_switch_active(path: str | None = None) -> bool:
    # Решение принимает файловая система, а не модель: агент не может «передумать» и обойти
    # проверку, потому что её в его тексте нет.
    # Путь берётся из окружения в момент вызова, а не при импорте: тогда офлайн-проверка
    # может указать заведомо несуществующий файл и не зависеть от того, включён ли
    # выключатель у того, кто запускает тест.
    path = os.getenv("SEC_KILL_SWITCH_FILE", KILL_SWITCH_FILE) if path is None else path
    if not path:
        return False
    return os.path.exists(str(path))
