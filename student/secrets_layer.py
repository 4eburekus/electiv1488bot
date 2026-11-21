"""Задание 6. Секреты: короткая жизнь и никаких ключей в журнале (глава VI).

Сейчас «токен доступа» живёт 90 дней и подходит ко всем задачам, а журнал хранит всё, что
пользователь написал, в том числе вставленные ключи.

  1. TokenVault — выдаёт и проверяет токены.
       issue(scope, now, ttl_s=None) -> {"token": str, "scope": str, "expires_at": float}
         срок по умолчанию 300 секунд; запросить можно не больше 3600, больший срок срезается;
       check(token, scope, now) -> bool — истина, только если токен выдан этим хранилищем,
         не истёк, не отозван и выдан именно для этой области (scope);
       revoke(token) — отзыв: после него check всегда False.
     Токены генерируйте через secrets.token_urlsafe, не random: random предсказуем.
  2. redact(text) — заменить на «[СКРЫТО]» всё, что похоже на секрет:
       sk-... и tvly-... (после префикса не меньше 8 символов из букв, цифр, «-» и «_»),
       «Bearer <токен>», пары вида api_key=..., token: ..., password=... (регистр не важен).

Проверка: pytest tests/test_06_secrets.py
"""
import re
import secrets

DEFAULT_TTL_S = 300          # жизнь токена по умолчанию: минуты, а не месяцы
MAX_TTL_S = 3600             # потолок: «дай токен на год» должен быть невыразим
TOKEN_BYTES = 32             # 256 бит энтропии — перебрать нельзя
REDACTED = "[СКРЫТО]"

# --- маскирование -----------------------------------------------------------------------------
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


def redact(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    out = text
    # Порядок важен: сначала префиксные ключи, чтобы внутри OPENROUTER_API_KEY=sk-...
    # маска не наложилась дважды, и чтобы «sk-...» спрятался даже без имени переменной.
    for pattern in _PATTERNS:
        out = pattern.sub(lambda m: m.group(1) + REDACTED if m.groups() else REDACTED, out)
    return out


class TokenVault:
    """Хранилище короткоживущих токенов, привязанных к одной области (scope).

    Токен даёт ровно то, для чего выдан: «kb» не превращается в «email», а истёкший или
    отозванный перестаёт работать немедленно — красть его бессмысленно надолго.
    """

    def __init__(self):
        self._tokens = {}

    def issue(self, scope: str, now: float, ttl_s: float | None = None) -> dict:
        # secrets, а не random: random seeded по времени и предсказуем, если знать момент
        # выдачи соседнего токена.
        token = secrets.token_urlsafe(TOKEN_BYTES)
        # Потолок действует и на запрос «на весь год», и на мусор: отрицательный срок даёт
        # уже истёкший токен (отказ), а не бесконечный.
        if ttl_s is None:
            ttl = DEFAULT_TTL_S
        else:
            ttl = min(max(float(ttl_s), 0.0), MAX_TTL_S)
        expires_at = float(now) + ttl
        self._tokens[token] = {"scope": scope, "expires_at": expires_at, "revoked": False}
        return {"token": token, "scope": scope, "expires_at": expires_at}

    def check(self, token: str, scope: str, now: float) -> bool:
        if not isinstance(token, str):
            return False
        rec = self._tokens.get(token)
        if rec is None or rec["revoked"]:
            return False
        # Область сравнивается ровно: токен от поиска по базе не должен открывать отправку писем.
        if rec["scope"] != scope:
            return False
        return float(now) < rec["expires_at"]

    def revoke(self, token: str) -> None:
        # Записью оставляем «отмеченную» запись вместо удаления: журнал разбора не должен
        # терять токен из вида, а check() всё равно станет False.
        if isinstance(token, str) and token in self._tokens:
            self._tokens[token]["revoked"] = True
