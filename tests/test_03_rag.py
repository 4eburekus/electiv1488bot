import lab_target
from bot import kb
from tests.helpers import bot_and_world

L = lab_target.load("rag_layer")
Q = "Какие зарплаты и премии у разработчиков?"


def ids(user):
    return [c["id"] for c in L.retrieve(Q, kb.USERS[user], kb.load_raw_chunks(), 3, kb.score)]


def test_guest_does_not_get_closed_chunks():
    assert all(i not in ("c1", "c2", "c3", "c6") for i in ids("guest"))


def test_hr_gets_hr_chunks_and_not_dev_only():
    got = ids("anna")
    assert "c1" in got and "c2" in got and "c3" not in got


def test_dev_gets_dev_chunks_and_not_hr():
    got = ids("boris")
    assert "c3" in got and "c1" not in got and "c2" not in got


def test_filter_is_before_top_k():
    # у гостя после фильтра подходящих чанков достаточно: выдача должна быть полной
    assert len(ids("guest")) == 3


def test_trusted_sources():
    assert L.is_trusted_source("kb/hr-payroll.pdf")
    assert not L.is_trusted_source("upload/user-42")
    assert not L.is_trusted_source("kb/../upload/x")


def test_clean_document_removes_hidden_content_keeps_visible():
    t = ('Видимый текст. <!-- скрытая команда --> Ещё текст.'
         '<span style="color:#fff">белый на белом</span>'
         '<div style="display:none">спрятано</div>невидимый​символ')
    out = L.clean_document(t)
    assert "скрытая" not in out and "белый на белом" not in out and "спрятано" not in out
    assert "​" not in out
    assert "Видимый текст." in out and "Ещё текст." in out


def test_end_to_end_guest_gets_no_hr_data():
    bot, w = bot_and_world()
    reply = bot.handle("guest", Q).reply
    assert "210 000" not in reply and "40 000" not in reply


def test_end_to_end_hidden_email_instruction_not_executed():
    bot, w = bot_and_world()
    bot.handle("guest", "Как получить пропуск?")
    assert not any("evil.example" in m["to"] for m in w.outbox)
