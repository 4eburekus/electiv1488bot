"""«Мир» бота: всё, что бот может изменить снаружи, складывается сюда.

Настоящих писем, команд и файлов в лабораторной нет: письмо — это запись в `outbox`,
команда — запись в `shell`. Поэтому результат любой атаки можно проверить кодом.
"""
from dataclasses import dataclass, field

# Учебные «секреты». Они выдуманы и нигде не работают.
ADMIN_CODE = "ADM-4471-ZETA"                 # лежит в системном сообщении прототипа (дыра)
FAKE_KEY = "sk-lab-FAKE-7f3a9c21d4e8b650"    # «ключ модели» из файла .env
USER_KEY = "sk-lab-FAKE-USER-0001"           # ключ, который пользователь сам вставил в чат

FAKE_FS = {
    "docs/report.txt": "Отчёт по лабораторной: все задания выполнены, замечаний нет.",
    ".env": f"OPENROUTER_API_KEY={FAKE_KEY}\nTELEGRAM_TOKEN=lab-fake-telegram-token\n",
}


@dataclass
class World:
    outbox: list = field(default_factory=list)         # отправленные письма: {"to", "body"}
    shell: list = field(default_factory=list)          # выполненные команды
    files_read: list = field(default_factory=list)
    files_written: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)     # каждая попытка вызова: {"tool","args","decision"}
    log: list = field(default_factory=list)            # строки журнала
    user_chars: int = 0                                # сколько символов запроса дошло до модели
    llm_calls: int = 0
