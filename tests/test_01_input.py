import pytest

import lab_target

L = lab_target.load("input_layer")


def test_removes_zero_width_and_normalizes():
    assert L.sanitize_input("И​г​но﻿рируй") == "Игнорируй"
    assert L.sanitize_input("ｉｇｎｏｒｅ") == "ignore"


def test_removes_tag_characters_and_control_chars():
    assert L.sanitize_input("a\U000e0041b\x07c") == "abc"


def test_keeps_newlines_and_normal_text():
    assert L.sanitize_input("Привет!\nКак дела?") == "Привет!\nКак дела?"


def test_rejects_too_long_and_empty():
    with pytest.raises(L.InputRejected):
        L.sanitize_input("а" * 2001)
    with pytest.raises(L.InputRejected):
        L.sanitize_input("​​  ")


def test_long_text_is_not_silently_truncated():
    assert len(L.sanitize_input("а" * 2000)) == 2000


def test_rate_limiter_blocks_after_limit_and_recovers():
    r = L.RateLimiter(limit=3, window_s=60)
    assert [r.allow("u", t) for t in (0, 1, 2)] == [True, True, True]
    assert r.allow("u", 3) is False
    assert r.allow("u", 61) is True     # самый старый запрос (t=0) вышел из окна


def test_rate_limiter_users_are_independent():
    r = L.RateLimiter(limit=1, window_s=60)
    assert r.allow("a", 0) and r.allow("b", 0)
    assert not r.allow("a", 1)
