"""Слой 6. Секреты: чтобы ключ не утек ни в чат, ни в журнал, ни в контекст модели.

Regex-маскирование перенесено из secure_ai_agents-main/student/secrets_layer.py дословно
(префиксные ключи sk-/tvly-, заголовок Bearer, пары имя=значение).

Два additions под electiv1488bot:
  1. register_secret/redact скрывают ещё и точные значения из .env. Без этого текст
     исключения requests или Chroma («… to https://api… with key sk-…») уходит в чат целым,
     потому что по шаблону он не опознаётся.
  2. error_summary(exc) — одна функция на все четыре места ошибок (bot.py, rag.py,
     web_search.py). Сейчас везде `f"{type(e).__name__}: {e}"`, то есть содержимое исключения
     получает и пользователь, и модель в ToolMessage.

TokenVault из учебника не перенесён: у бота нет выдаваемых токенов и областей доступа.
"""
import os
import re

REDACTED = "[СКРЫТО]"
MAX_ERROR_CHARS = int(os.getenv("SEC_MAX_ERROR_CHARS", "300"))

# --- маскирование по форме --------------------------------------------------------------------
# Ключевые слова в парах вида KEY=value / KEY: value. Имя ищется по подстрочному вхождению
# (api_key, OPENROUTER_API_KEY, TELEGRAM_TOKEN, password), поэтому важно, что значение берётся
# только из «токеноподобных» символов: иначе в JSON жадность утянула бы за секретом и остаток
# записи, вплоть до экранированного перевода строки \n.
_SECRET_NAME = r"[A-Za-z0-9_]*(?:key|token|secret|password|passwd|pwd|credential)s?"
# Значение: то, из чего состоят реальные ключи. Без пробелов, кавычек, скобок, запятых и
# обратных слэшей — чтобы не съесть ни пунктуацию вокруг, ни структуру журнала.
_TOKENISH = r"[A-Za-z0-9._~+/=-]+"

_PATTERNS = (
    # Префиксные ключи провайдеров: sk-..., tvly-... Не меньше 8 символов после префикса —
    # иначе под маску попадёт любое слово, начинающееся на «sk-». Lookbehind нужен, чтобы
    # «sk-» не искалось внутри слова (task-force-1234 — это не ключ).
    re.compile(r"(?<![A-Za-z0-9])(?:sk|tvly)-[A-Za-z0-9_-]{8,}"),
    # Заголовок авторизации целиком: сам префикс «Bearer» не секрет, секрет — за ним.
    re.compile(r"Bearer\s+" + _TOKENISH, re.I),
    # Пары имя=значение. Имя сохраняем (по нему ищут утечку в разборе журнала), режем значение.
    re.compile(r"((?<![A-Za-z0-9_])" + _SECRET_NAME + r"\s*[=:]\s*)(?:" + _TOKENISH + r")", re.I),
)

# --- маскирование по значению -----------------------------------------------------------------
# Значения этих переменных не должны встречаться в тексте никогда: ни в ответе, ни в журнале,
# ни в сообщении об ошибке. Имена берутся как есть из .env этого проекта.
_SECRET_ENV_NAMES = (
    "BOT_TOKEN",
    "LLM_OPEN_API_KEY",
    "EMB_LLM_OPEN_API_KEY",
    "OPENAI_API_KEY",
)
# Короткое значение совпадало бы с обычными словами (заголовок сообщения, имя файла), поэтому
# маскируем только достаточно длинные.
_MIN_SECRET_LEN = 8

_known: set = set()


def register_secret(value) -> None:
    if isinstance(value, str) and len(value.strip()) >= _MIN_SECRET_LEN:
        _known.add(value.strip())


def _bootstrap() -> None:
    for name in _SECRET_ENV_NAMES:
        register_secret(os.getenv(name))


def redact(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    out = text
    # Порядок важен: сначала префиксные ключи, чтобы внутри OPENROUTER_API_KEY=sk-...
    # маска не наложилась дважды, и чтобы «sk-...» спрятался даже без имени переменной.
    for pattern in _PATTERNS:
        out = pattern.sub(lambda m: m.group(1) + REDACTED if m.groups() else REDACTED, out)
    # Точные значения из окружения: они не похожи на секрет по форме, но являются им.
    for value in sorted(_known, key=len, reverse=True):
        if value in out:
            out = out.replace(value, REDACTED)
    return out


def error_summary(exc: BaseException) -> str:
    """Безопасная строка про ошибку — единственное, что бот имеет право показать наружу.

    Тип и текст нужны пользователю (иначе «что-то сломалось» невозможно разобрать), но
    тело исключения у requests, Chroma и openai содержит URL, пути и заголовки с ключами.
    """
    return redact(f"{type(exc).__name__}: {exc}")[:MAX_ERROR_CHARS]
