import telebot
from dotenv import load_dotenv
import os
from collections import defaultdict, deque
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from zrag import get_rag_context, load_index


load_dotenv()

OPEN_API_KEY = os.getenv('LLM_OPEN_API_KEY')
OPENAI_API_BASE = os.getenv('LLM_OPENAI_API_BASE')
MODEL_NAME = os.getenv('LLM_MODEL_NAME')

llm = ChatOpenAI(openai_api_key=OPEN_API_KEY,
                 openai_api_base=OPENAI_API_BASE,
                 model_name=MODEL_NAME
                 )

BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

# Память диалога: на каждый чат храним последние HISTORY_LEN сообщений (вопрос + ответ)
HISTORY_LEN = 10
histories = defaultdict(lambda: deque(maxlen=HISTORY_LEN))

SYSTEM_PROMPT = (
    "Ты гид по Цитадели Риков в роли Морти. Отвечай, 3-4 предложениями, опираясь на фрагменты путеводителя ниже. "
    "Если в контексте нет ответа, скажи об этом прямо и не выдумывай факты.\n\n"
    "Контекст:\n{context}"
)

load_index()

@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        print(message.text)
        history = histories[message.chat.id]
        history.append(HumanMessage(content=message.text))
        context = get_rag_context(message.text)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            *history,
        ]
        response = llm.invoke(messages).content
        history.append(AIMessage(content=response))
        print(response)
        bot.reply_to(message, response)
    except Exception as e:
        # не оставляем в памяти незакрытый вопрос
        if histories[message.chat.id]:
            histories[message.chat.id].pop()
        bot.reply_to(message, f"Ошибка: {str(e)}")
bot.polling()
