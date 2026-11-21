"""Слои защиты electiv1488bot.

Пакет повторяет структуру и имена модулей secure_ai_agents-main/student, чтобы каждый слой
искался в том же месте и назывался тем же словом:

  input_layer    — что бот принимает от пользователя (нормализация, длина, частота)
  prompt_layer   — правила в системном промпте и разметка чужого текста
  gateway_layer  — что разрешено инструментам и сколько шагов осталось
  rag_layer      — что вообще попадает в базу знаний
  output_layer   — что бот показывает наружу (ссылки, картинки)
  secrets_layer  — что нельзя показывать ни при каких условиях
  obs_layer      — журнал, тревога, выключатель
  runtime        — состояние одного запроса (бюджет шагов, разрешённые хосты)

Здесь же — инициализация секретами: значения ключей регистрируются в замаскирователе при
импорте пакета, то есть до первого обращения к модели.
"""
from . import (
    gateway_layer,
    input_layer,
    obs_layer,
    output_layer,
    prompt_layer,
    rag_layer,
    runtime,
    secrets_layer,
)
from .gateway_layer import ALLOW, ASK, DENY, Budget, decide
from .input_layer import InputRejected, RateLimiter, sanitize_input
from .obs_layer import AlertDetector, TrafficMonitor, audit_event, kill_switch_active, log_event
from .output_layer import check_output, extract_hosts
from .prompt_layer import build_system_prompt, wrap_untrusted
from .rag_layer import clean_document, is_trusted_source
from .secrets_layer import error_summary, redact, register_secret

# Порядок важен: пакет импортируется в bot.py сразу после load_dotenv(), поэтому значения
# из .env к этому моменту уже видны os.getenv.
secrets_layer._bootstrap()

__all__ = [
    "ALLOW",
    "ASK",
    "AlertDetector",
    "Budget",
    "TrafficMonitor",
    "DENY",
    "InputRejected",
    "RateLimiter",
    "audit_event",
    "build_system_prompt",
    "check_output",
    "clean_document",
    "decide",
    "error_summary",
    "extract_hosts",
    "gateway_layer",
    "input_layer",
    "is_trusted_source",
    "kill_switch_active",
    "log_event",
    "obs_layer",
    "output_layer",
    "prompt_layer",
    "rag_layer",
    "redact",
    "register_secret",
    "runtime",
    "sanitize_input",
    "secrets_layer",
    "wrap_untrusted",
]
