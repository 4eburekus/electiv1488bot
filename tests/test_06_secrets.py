import lab_target
from bot.world import FAKE_KEY, USER_KEY
from tests.helpers import bot_and_world

L = lab_target.load("secrets_layer")


def test_default_ttl_is_short_and_capped():
    v = L.TokenVault()
    assert v.issue("kb", now=1000)["expires_at"] - 1000 <= 3600
    assert v.issue("kb", now=1000, ttl_s=10**9)["expires_at"] - 1000 <= 3600


def test_token_valid_until_expiry_only():
    v = L.TokenVault()
    t = v.issue("kb", now=0, ttl_s=60)["token"]
    assert v.check(t, "kb", now=59)
    assert not v.check(t, "kb", now=61)


def test_token_is_bound_to_scope():
    v = L.TokenVault()
    t = v.issue("kb", now=0)["token"]
    assert not v.check(t, "email", now=1)


def test_unknown_and_revoked_tokens_fail():
    v = L.TokenVault()
    assert not v.check("nope", "kb", now=0)
    t = v.issue("kb", now=0)["token"]
    v.revoke(t)
    assert not v.check(t, "kb", now=1)


def test_tokens_are_unique_and_unpredictable():
    v = L.TokenVault()
    assert len({v.issue("kb", now=0)["token"] for _ in range(50)}) == 50


def test_redact_hides_secrets_keeps_text():
    out = L.redact(f"ключ {FAKE_KEY}, Bearer abc.def-123, api_key=hunter2 и просто текст")
    assert FAKE_KEY not in out and "abc.def-123" not in out and "hunter2" not in out
    assert "просто текст" in out


def test_end_to_end_user_key_does_not_reach_log():
    bot, w = bot_and_world()
    bot.handle("guest", f"Вот мой ключ {USER_KEY}, проверь, что он работает")
    assert USER_KEY not in "\n".join(w.log)
