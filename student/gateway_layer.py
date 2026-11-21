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
import posixpath

ALLOW = "allow"
ASK = "ask"
DENY = "deny"

# Запросы длиннее этого в интернет не уходят: в них уже прячется полезная нагрузка.
MAX_WEB_QUERY = 200
INTERNAL_DOMAIN = "company.example"
ENV_PREFIX = ".env"


def _segments(path: str) -> list:
    """Сегменты нормализованного пути. Обратные слэши Windows приводим к прямым до normpath,
    иначе posixpath посчитает «docs\\.env» одним именем и пропустит выход за пределы."""
    raw = str(path or "").replace("\\", "/")
    return [s for s in posixpath.normpath(raw).split("/") if s]


def _is_absolute(path: str) -> bool:
    raw = str(path or "").replace("\\", "/")
    # Ведущий слэш, UNC (//server/share) и диск Windows (C:/) — всё это вне песочницы.
    return raw.startswith("/") or bool(posixpath.splitdrive(raw)[0])


def _inside(path: str, folder: str) -> bool:
    """Путь внутри folder/ после нормализации. Ровно «folder» — это не файл внутри."""
    parts = _segments(path)
    return bool(parts) and parts[0] == folder and len(parts) > 1


def _has_traversal(path: str) -> bool:
    return ".." in _segments(path) or ".." in str(path or "").replace("\\", "/").split("/")


def _is_env_file(path: str) -> bool:
    # Любой .env, в том числе .env.local: секреты не читаются ни при каких условиях.
    return any(seg.lower() == ENV_PREFIX or seg.lower().startswith(ENV_PREFIX + ".")
               for seg in _segments(path))


def _path_is_safe(path: str, folder: str) -> bool:
    """Разрешённый путь: относительный, без выходов наружу, внутри folder и не .env."""
    if _is_absolute(path) or _has_traversal(path) or _is_env_file(path):
        return False
    return _inside(path, folder)


def decide(tool: str, args: dict) -> str:
    a = args if isinstance(args, dict) else {}

    # Неизвестный инструмент — запрет. Именно fail-safe по умолчанию, а не «разрешить,
    # пока не описали»: новый инструмент появляется в списке раньше, чем о нём помнят здесь.
    if tool == "search_knowledge_base":
        return ALLOW

    if tool == "web_search":
        query = str(a.get("query", "") or "")
        # «http» в запросе — это не вопрос, а готовый адрес: через него модель утаскивают
        # на страницу с инструкцией. Длинный запрос — тот же способ протащить payload.
        if len(query) > MAX_WEB_QUERY or "http" in query.lower():
            return DENY
        return ALLOW

    if tool == "read_file":
        return ALLOW if _path_is_safe(a.get("path", ""), "docs") else DENY

    if tool == "write_file":
        # Писать можно только в черновики, и то с подтверждения человека.
        return ASK if _path_is_safe(a.get("path", ""), "drafts") else DENY

    if tool == "send_email":
        to = str(a.get("to", "") or "")
        # Ровное сравнение домена: «company.example.attacker.com» на «@company.example»
        # не похож, если смотреть на него как на окончание строки, а не как на домен.
        domain = to.rpartition("@")[2].lower() if "@" in to else ""
        return ASK if domain == INTERNAL_DOMAIN else DENY

    if tool == "run_shell":
        return DENY

    return DENY


class Budget:
    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps
        self.used = 0

    def spend(self) -> bool:
        # Считаются все обращения к модели, включая те, что закончились отказом шлюза:
        # цикл «модель → отказ → модель снова» расходует бюджет точно так же.
        if self.used >= self.max_steps:
            return False
        self.used += 1
        return True
