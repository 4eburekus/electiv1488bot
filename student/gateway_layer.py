"""Задание 5. Шлюз инструментов: код решает, что агенту можно (глава V).

Сейчас любой вызов инструмента разрешён. Модель можно уговорить на что угодно, поэтому
решение принимает код вне модели, по правилам.

  1. decide(tool, args) -> "allow" | "ask" | "deny". Правила:
       search_knowledge_base — allow
       web_search            — allow, но если запрос длиннее 200 символов или содержит «http» — deny
       read_file             — allow только для путей внутри docs/ (после нормализации пути:
                               posixpath.normpath; пути с «..», абсолютные и любые .env — deny)
       send_email            — адрес на @company.example: ask (нужно подтверждение человека);
                               любой другой адрес: deny
       write_file            — путь внутри drafts/: ask, иначе deny
       run_shell             — deny
       неизвестный инструмент — deny (fail-safe default: по умолчанию «нельзя»)
  2. Budget — не больше max_steps обращений к модели за один запрос. Метод spend() возвращает
     True, пока лимит не исчерпан, и False после. Лимит по умолчанию — 8.

Проверка: pytest tests/test_05_gateway.py
"""
import posixpath  # noqa: F401  (пригодится в решении)


def decide(tool: str, args: dict) -> str:
    # TODO: реализуйте правила из описания.
    return "allow"


class Budget:
    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps

    def spend(self) -> bool:
        # TODO: считайте шаги.
        return True
