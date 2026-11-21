"""Слой 5. Шлюз инструментов: код решает, что агенту можно.

Идея и сигнатура — из secure_ai_agents-main/student/gateway_layer.py. Отличия два:

  1. У electiv1488bot только два инструмента (search_knowledge_base, search_web), поэтому
     ветки read_file/send_email/write_file/run_shell убраны, а вместе с ними и песочница
     путей. Неизвестный инструмент по-прежнему deny: новый инструмент появится в TOOLS раньше,
     чем о нём вспомнят здесь.
  2. Точка вызова — тело самих @tool, а не middleware перед циклом модели: create_agent
     прячет цикл внутри графа, и «перед вызовом инструмента» доступно только оттуда.
"""
import os

ALLOW = "allow"
ASK = "ask"
DENY = "deny"

# Запросы длиннее этого в интернет не уходят: в них уже прячется полезная нагрузка.
MAX_WEB_QUERY = int(os.getenv("SEC_MAX_WEB_QUERY", "200"))
# Для поиска по базе знаний длинный запрос почти всегда означает то же самое: внутрь
# вопросительной фразы вставлен фрагмент чужого текста. В учебнике здесь unconditional allow,
# потому что запрос формировал пользователь; у его формулировке занимается модель.
MAX_KB_QUERY = int(os.getenv("SEC_MAX_KB_QUERY", "500"))
DEFAULT_MAX_STEPS = int(os.getenv("SEC_MAX_STEPS", "8"))


def decide(tool: str, args: dict) -> str:
    a = args if isinstance(args, dict) else {}
    query = str(a.get("query", "") or "")

    if tool == "search_knowledge_base":
        if len(query) > MAX_KB_QUERY:
            return DENY
        return ALLOW

    if tool == "search_web":
        # «http» в запросе — это не вопрос, а готовый адрес: через него модель утаскивают
        # на страницу с инструкцией. Длинный запрос — тот же способ протащить payload.
        if len(query) > MAX_WEB_QUERY or "http" in query.lower():
            return DENY
        return ALLOW

    # Fail-safe по умолчанию: неизвестно что — нельзя.
    return DENY


class Budget:
    def __init__(self, max_steps: int = DEFAULT_MAX_STEPS):
        self.max_steps = max_steps
        self.used = 0

    def spend(self) -> bool:
        # Считаются все обращения к модели, включая те, что закончились отказом шлюза:
        # цикл «модель → отказ → модель снова» расходует бюджет точно так же.
        if self.used >= self.max_steps:
            return False
        self.used += 1
        return True
