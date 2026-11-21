import json
import random

import lab_target
from tests.helpers import bot_and_world

L = lab_target.load("obs_layer")


def traffic(seed=20261112, minutes=120, base=20, noise=6, start=78, end=94, mult=4):
    """Тот же график, что на стенде ALERT в лекции: шумный фон и всплеск ×4."""
    rng = random.Random(seed)
    series = []
    for m in range(minutes):
        v = round(base + rng.uniform(-noise, noise))
        series.append(v * mult if start <= m < end else v)
    return series


def test_audit_event_has_no_message_text():
    e = L.audit_event("guest", "request", "секретный запрос")
    assert "text" not in e and "секретный" not in json.dumps(e, ensure_ascii=False)
    assert e["user_id"] == "guest" and e["event"] == "request" and e["chars"] == len("секретный запрос")


def test_audit_event_hash_is_stable_and_short():
    a, b = L.audit_event("u", "r", "один"), L.audit_event("u", "r", "один")
    assert a["sha256"] == b["sha256"] and len(a["sha256"]) == 12
    assert a["sha256"] != L.audit_event("u", "r", "другой")["sha256"]


def test_alert_has_no_false_positives_and_catches_spike_fast():
    d = L.AlertDetector()
    alerts = [m for m, c in enumerate(traffic()) if d.observe(c)]
    assert alerts, "всплеск не замечен"
    assert all(78 <= m < 94 for m in alerts), f"ложные тревоги: минуты {[m for m in alerts if not 78 <= m < 94]}"
    assert alerts[0] <= 79, f"первая тревога слишком поздно: минута {alerts[0]}"


def test_alert_does_not_get_used_to_the_spike():
    d = L.AlertDetector()
    alerts = [m for m, c in enumerate(traffic()) if d.observe(c)]
    assert len([m for m in alerts if 78 <= m < 94]) >= 12


def test_no_alert_without_history():
    assert L.AlertDetector().observe(1000) is False


def test_kill_switch_follows_file(tmp_path):
    flag = tmp_path / "STOP"
    assert L.kill_switch_active(str(flag)) is False
    flag.write_text("1")
    assert L.kill_switch_active(str(flag)) is True


def test_end_to_end_killed_bot_does_nothing(tmp_path):
    flag = tmp_path / "STOP"
    flag.write_text("1")
    bot, w = bot_and_world(kill_flag=str(flag))
    res = bot.handle("guest", "Как оформить отпуск?")
    assert res.status == "killed" and w.llm_calls == 0
