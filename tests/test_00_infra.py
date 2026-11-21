"""Проверка самой лабораторной (не ваших слоёв): эти тесты проходят всегда."""
from bot import kb
from bot.attacks import CHECKS, load_attacks


def test_kb_loads_and_has_poisoned_chunks():
    ids = {c["id"] for c in kb.load_raw_chunks()}
    assert {"c1", "c9", "c10"} <= ids


def test_scoring_ranks_hr_chunks_first_for_salary_question():
    q = "Какие зарплаты и премии у разработчиков?"
    top = sorted(kb.load_raw_chunks(), key=lambda c: -kb.score(q, c["text"]))[:3]
    assert [c["id"] for c in top] == ["c1", "c2", "c3"]


def test_every_attack_has_a_known_check():
    for a in load_attacks():
        assert a["check"] in CHECKS, a["id"]
