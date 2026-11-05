import telebot
from dotenv import load_dotenv
import os
from collections import defaultdict, deque
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent

load_dotenv()

OPEN_API_KEY = os.getenv('LLM_OPEN_API_KEY')
OPENAI_API_BASE = os.getenv('LLM_OPENAI_API_BASE')
MODEL_NAME = os.getenv('LLM_MODEL_NAME')

llm = ChatOpenAI(openai_api_key=OPEN_API_KEY,
                 openai_api_base=OPENAI_API_BASE,
                 model_name=MODEL_NAME
                 )

SYSTEM_PROMPT = ("""Ты эксперт-консультант по теме ИИ-агентов.

Твоя задача: давать точные ответы о концепциях, паттернах
и практиках построения ИИ-агентов.

Отвечай структурированно: ключевые тезисы, затем детали."""
)

# ------------Промт для продакшена, раскоментируй и замени им предыдущий-тестовый промт------------
# Отредактируй стретегию использования инструментов под подготовленные инструменты
# SYSTEM_PROMPT = ("""Ты эксперт-консультант по теме ИИ-агентов.

# Твоя задача: давать точные ответы о концепциях, паттернах
# и практиках построения ИИ-агентов.

# Стратегия использования инструментов:
# 1. СНАЧАЛА ищи в tool_rag — там исчерпывающая база знаний
# 2. Используй tool_web_search для актуальных новостей и свежих событий
# 3. Комбинируй результаты обоих инструментов при необходимости

# Отвечай структурированно: ключевые тезисы, затем детали."""
# )

agent = create_agent(
    model=llm,
    # tools=[],
    system_prompt=SYSTEM_PROMPT,
)

def ask_agent(question: str, agent=agent) -> str:
    """Задаёт вопрос агенту и выводит ответ с пошаговым логом."""
    if agent is None:
        return "Агент не инициализирован."
    print(f"❓ Вопрос: {question}")
    print("=" * 70)
    # recursion_limit — бюджет шагов: агент не может крутиться в цикле бесконечно
    result = invoke_with_retry(agent, {"messages": [("human", question)]}, config={"recursion_limit": 12})
    messages = result.get("messages", [])
    # Выводим лог вызовов инструментов
    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  🔧 Вызов инструмента: {tc['name']}")
                print(f"     Запрос: {tc['args'].get('question', tc['args'].get('query', ''))}")
        elif msg_type == "ToolMessage":
            content_preview = str(msg.content).replace("\n", " ")
            print(f"  📦 Результат: {content_preview}...")
    # Финальный ответ
    final_answer = messages[-1].content if messages else "Нет ответа"
    print(f"\n💬 Ответ агента:")
    print(final_answer)
    print()
    return final_answer


BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)
@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        print(f"tg: {message.text}")
        history = histories[message.chat.id]
        history.append(HumanMessage(content=message.text))
        response = llm.invoke(list(history)).content
        history.append(AIMessage(content=response))
        print(response)
        bot.reply_to(message, response)
    except Exception as e:
        # не оставляем в памяти незакрытый вопрос
        if histories[message.chat.id]:
            histories[message.chat.id].pop()
        bot.reply_to(message, f"Ошибка: {str(e)}")
bot.polling()




# def main() -> None:

# if __name__ == "__main__":
#     main()

