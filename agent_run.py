"""Раздел 11 ноутбука: ИИ-агент-консультант на локальных моделях (без облачных ключей).

Архитектура как в разделе 11, только источники — локальные:
  пользователь → langchain.agents.create_agent (ReAct)
                   ├─ search_knowledge_base → FAISS + реранкер + create_retrieval_chain
                   └─ web_search            → DuckDuckGo
  LLM и эмбеддинги — с локального OpenAI-совместимого эндпоинта (LM Studio / gpustack).

Запуск (переменные берутся из .env.agent, иначе из .env):
    myvenv\\Scripts\\python.exe agent_run.py check
    myvenv\\Scripts\\python.exe -X utf8 agent_run.py eval
    myvenv\\Scripts\\python.exe -X utf8 agent_run.py tests
    myvenv\\Scripts\\python.exe -X utf8 agent_run.py ask "Что такое ReAct паттерн?"
    myvenv\\Scripts\\python.exe -X utf8 agent_run.py           # интерактивный режим
Индекс базы знаний собирается отдельно:
    myvenv\\Scripts\\python.exe -X utf8 kb_index.py build
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
preferred = BASE / ".env.agent"
load_dotenv(preferred if preferred.exists() else BASE / ".env")

sys.path.insert(0, str(BASE))

import agent  # noqa: E402  (после загрузки окружения — модуль читает переменные при импорте)

if __name__ == "__main__":
    sys.exit(agent.main(sys.argv[1:]))
