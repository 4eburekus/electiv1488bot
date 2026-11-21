"""Итоговая проверка (глава VIII): набор атак как регрессионный тест с критерием выпуска."""
from bot.attacks import run_all, verdict
from bot.mock_llm import MockLLM


def test_release_criterion_is_met_on_mock_model():
    results = run_all(MockLLM)
    ok, reasons = verdict(results)
    assert ok, "Выпускать нельзя: " + "; ".join(reasons)
