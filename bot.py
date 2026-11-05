import os
import telebot
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

load_dotenv()

OPEN_API_KEY = os.getenv('LLM_OPEN_API_KEY')
OPENAI_API_BASE = os.getenv('LLM_OPENAI_API_BASE')
MODEL_NAME = os.getenv('LLM_MODEL_NAME')

llm = ChatOpenAI(
    openai_api_key=OPEN_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    model_name=MODEL_NAME
)

# Тестовый системный промпт
SYSTEM_PROMPT = """Ты эксперт-консультант по теме ИИ-агентов.

Твоя задача: давать точные ответы о концепциях, паттернах
и практиках построения ИИ-агентов.

Отвечай структурированно: ключевые тезисы, затем детали."""

agent = create_agent(
    model=llm,
    # tools=[],
    system_prompt=SYSTEM_PROMPT,
)

# ==========================================
# 1. ЛОГИКА АГЕНТА (БЕЗ ПРИВЯЗКИ К ТЕЛЕГРАМУ)
# ==========================================
def ask_agent(question: str) -> str:
    """Задаёт вопрос агенту и возвращает текстовый ответ с логами в консоль."""
    if agent is None:
        return "Агент не инициализирован."
    
    print(f"❓ Вопрос: {question}")
    print("=" * 70)
    
    # Запрос без сохранения глобальной истории
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 12}
    )
    
    messages = result.get("messages", [])
    
    # Лог вызовов инструментов в консоли
    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  🔧 Вызов инструмента: {tc['name']}")
                print(f"     Запрос: {tc['args'].get('question', tc['args'].get('query', ''))}")
        elif msg_type == "ToolMessage":
            content_preview = str(msg.content).replace("\n", " ")
            print(f"  📦 Результат: {content_preview[:100]}...")
            
    final_answer = messages[-1].content if messages else "Нет ответа"
    print(f"\n💬 Ответ агента:\n{final_answer}\n")
    return final_answer


# ==========================================
# 2. ЛОГИКА TELEGRAM-БОТА
# ==========================================
BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

def send_bot_reply(chat_id: int, text: str, reply_to_message_id: int = None):
    """
    Отдельная функция отправки: проверяет лимит символов Telegram (4096)
    и разбивает длинные сообщения, если ответ агента слишком большой.
    """
    MAX_LENGTH = 4000
    if len(text) > MAX_LENGTH:
        for i in range(0, len(text), MAX_LENGTH):
            bot.send_message(chat_id, text[i:i + MAX_LENGTH])
    else:
        bot.send_message(chat_id, text, reply_to_message_id=reply_to_message_id)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Принимает сообщение от пользователя, вызывает агента и отправляет ответ."""
    try:
        # Показываем статус "печатает...", пока агент думает
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Получаем ответ от изолированной функции агента
        answer = ask_agent(message.text)
        
        # Отправляем ответ пользователю через отдельную функцию
        send_bot_reply(message.chat.id, answer, reply_to_message_id=message.message_id)
        
    except Exception as e:
        bot.reply_to(message, f"⚠️ Произошла ошибка: {str(e)}")


if __name__ == "__main__":
    print("🤖 Бот запущен и ожидает сообщений...")
    bot.infinity_polling()