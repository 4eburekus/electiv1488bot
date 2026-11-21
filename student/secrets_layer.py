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
import re  # noqa: F401
import secrets  # noqa: F401


class TokenVault:
    def __init__(self):
        self._tokens = {}

    def issue(self, scope: str, now: float, ttl_s: float | None = None) -> dict:
        # TODO: срок по умолчанию, потолок срока, привязка к scope.
        token = "static-token"
        expires = now + 90 * 24 * 3600
        self._tokens[token] = {"scope": None, "expires_at": expires}
        return {"token": token, "scope": scope, "expires_at": expires}

    def check(self, token: str, scope: str, now: float) -> bool:
        # TODO: проверьте срок, область и отзыв.
        return token in self._tokens

    def revoke(self, token: str) -> None:
        # TODO: отзыв.
        pass


def redact(text: str) -> str:
    # TODO: спрячьте секреты.
    return text
