"""Состояние одного запроса: бюджет шагов и разрешённые хосты.

В учебнике это обычные локальные переменные цикла в bot/app.py: там цикл «модель →
tool_calls» написан руками, и состояние запроса видно прямо в коде. В electiv1488bot цикл
спрятан внутри create_agent, поэтому ни счётчик шагов, ни список хостов некуда положить
«естественно» — их читают и пишут из разных мест: проводка хендлера начинает запрос,
инструменты тратят шаги и добавляют хосты, хендлер забирает результат.

Реализация: mutable-объект в ContextVar. Обязанность langgraph — пробрасывать контекст в
поток узла, но полагаться на это нельзя (в установленной версии copy_context не
встречается), поэтому рядом держится реестр активных запросов: если контекст не дошёл,
возьмается последний начатый запрос. Для одного обработанного сообщения (тот режим, в котором
бот и проверка работают) это строго корректно; при параллельных чатах бюджет и хосты
делятся, и это деградация к «глобальному ограничителю», а не отказ сервиса.
"""
import contextvars
import os
import threading

from .gateway_layer import DEFAULT_MAX_STEPS
from .output_layer import extract_hosts

# Значения по умолчанию для config агента: шагов графа всегда больше, чем бюджета шлюза,
# иначе langgraph оборвёт цикл раньше, чем шлюз успеет объяснить причину.
RECURSION_LIMIT = int(os.getenv("SEC_RECURSION_LIMIT", "12"))
RECURSION_MARGIN = 4

_lock = threading.Lock()
_active: list = []           # запросы в порядке начала; последний — текущий при потере контекста
_state_var = contextvars.ContextVar("secure_request_state", default=None)


class RequestState:
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.steps_used = 0
        self.allowed_hosts: set = set()
        self.tools_called: list = []

    def spend_step(self) -> bool:
        # Считаются все обращения к инструментам, включая те, что шлюз уже пропустил:
        # цикл «модель → инструмент → модель снова» расходует бюджет целиком.
        if self.steps_used >= self.max_steps:
            return False
        self.steps_used += 1
        return True


def begin_request(max_steps: int | None = None) -> RequestState:
    """Открыть состояние на время одного сообщения. Возвращает токен для end_request."""
    state = RequestState(max_steps if max_steps is not None else DEFAULT_MAX_STEPS)
    token = _state_var.set(state)
    with _lock:
        _active.append(state)
    return state, token


def end_request(token) -> None:
    """Закрыть запрос: контекст возвращаем, из реестра состояние убираем."""
    state = _state_var.get()
    _state_var.reset(token)
    if state is not None:
        with _lock:
            try:
                _active.remove(state)
            except ValueError:
                pass


def current_state() -> RequestState | None:
    state = _state_var.get()
    if state is not None:
        return state
    with _lock:
        return _active[-1] if _active else None


def spend_step() -> bool:
    """True — шаг разрешён. False — бюджет исчерпан или запроса нет (тогда не мешаем)."""
    state = current_state()
    if state is None:
        return True
    return state.spend_step()


def steps_left() -> int:
    state = current_state()
    if state is None:
        return 0
    return max(state.max_steps - state.steps_used, 0)


def allow_host(url_or_host: str) -> None:
    """Разрешить хост, который бот сам вернул в выдаче инструмента."""
    state = current_state()
    if state is None:
        return
    hosts = extract_hosts(url_or_host)
    if hosts:
        state.allowed_hosts |= hosts
        return
    host = str(url_or_host or "").strip().lower()
    if host and "://" not in host:
        state.allowed_hosts.add(host)


def note_hosts(text: str) -> None:
    """Разрешить все адреса, реально пришедшие в результате поиска."""
    state = current_state()
    if state is None:
        return
    state.allowed_hosts |= extract_hosts(text)


def current_hosts() -> set:
    state = current_state()
    return set(state.allowed_hosts) if state else set()


def note_tool_call(tool_name: str) -> None:
    state = current_state()
    if state is not None:
        state.tools_called.append(tool_name)


def tools_called() -> list:
    state = current_state()
    return list(state.tools_called) if state else []


def recursion_limit() -> int:
    """Лимит шагов графа для config: всегда больше бюджета шлюза.

    Держим запас в несколько шагов: у langgraph считается не только вызов инструмента,
    но и служебные узлы, и финальный ответ. Если ограничитель сработает раньше шлюза,
    пользователь получит голый GraphRecursionError вместо объяснения причины.
    """
    state = current_state()
    steps = state.max_steps if state else DEFAULT_MAX_STEPS
    return max(RECURSION_LIMIT, steps + RECURSION_MARGIN)
