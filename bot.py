import os
import time

import telebot
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

# load_dotenv() ДО импорта инструментов: rag.py читает EMB_LLM_* и RAG_* на уровне модуля,
# без .env он не сможет сверить модель эмбеддингов индекса
load_dotenv()

from tool_rag import rag
from tool_rag.rag import get_retrieve, search_knowledge_base
from tool_web_search.web_search import search_web

OPEN_API_KEY = os.getenv('LLM_OPEN_API_KEY')
OPENAI_API_BASE = os.getenv('LLM_OPENAI_API_BASE')
MODEL_NAME = os.getenv('LLM_MODEL_NAME')

llm = ChatOpenAI(
    openai_api_key=OPEN_API_KEY,
    openai_api_base=OPENAI_API_BASE,
    model_name=MODEL_NAME,
    # Без этого клиент ждёт ответ 600 секунд, и зависший роутер выглядит как умерший бот
    request_timeout=120,
    max_retries=1,
)

# Тестовый системный промпт (без инструментов) — оставлен на случай отладки без RAG/поиска
# SYSTEM_PROMPT = """Ты эксперт-консультант по теме ИИ-агентов.
#
# Твоя задача: давать точные ответы о концепциях, паттернах
# и практиках построения ИИ-агентов.
#
# Отвечай структурированно: ключевые тезисы, затем детали."""

# ------------Промт для продакшена------------
# Имена инструментов те, что реально подключены ниже в TOOLS
SYSTEM_PROMPT = ("""Ты эксперт-консультант по теме ИИ-агентов.
Твоя задача: давать точные ответы о концепциях, паттернах
и практиках построения ИИ-агентов.

Стратегия использования инструментов:
1. СНАЧАЛА ищи в search_knowledge_base — там учебная база знаний по ИИ-агентам
2. Используй search_web для актуальных новостей и свежих событий
3. Комбинируй результаты обоих инструментов при необходимости
4. Если инструмент ответил текстом, который начинается с ⚠️, честно скажи, что источник
   недоступен, — не выдумывай ответ вместо него

Отвечай структурированно: ключевые тезисы, затем детали.
Ссылайся на источники: файл и страницу из базы знаний, ссылку из веб-поиска.""")

TOOLS = [search_knowledge_base, search_web]

agent = create_agent(
    model=llm,
    tools=TOOLS,
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
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 12}
        )
    except Exception as e:
        # Без этого исключения Telegram-слой показал бы «бот молчит»
        print(f"❌ Агент не ответил: {type(e).__name__}: {e}")
        return f"⚠️ Не получилось получить ответ: {type(e).__name__}: {e}"

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

    final_answer = str(getattr(messages[-1], "content", "")) if messages else "Нет ответа"
    print(f"\n💬 Ответ агента:\n{final_answer}\n")
    return final_answer or "Агент вернул пустой ответ — попробуй переформулировать вопрос."


# ==========================================
# 2. ЛОГИКА TELEGRAM-БОТА
# ==========================================
BOT_TOKEN = os.getenv('BOT_TOKEN')
MAX_MESSAGE_LENGTH = 4000 # Ограничение сообщения tg
bot = telebot.TeleBot(BOT_TOKEN)


def split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Режет длинный ответ на части по границам строк, не разрывая блоки кода.

    Пустые строки сохраняем на стыках: они несут абзацы Markdown, и если их съесть,
    два тезиса склеятся в одну строку. Если разрыв попадает внутрь блока ``` —
    закрываем его в конце части и переоткрываем в начале следующей.
    """
    if len(text) <= limit:
        return [text]

    FENCE = "```"
    CLOSE = FENCE + "\n"     # то, что допишем в конец части, если блок кода не закрыт
    parts: list[str] = []
    buf: list[str] = []      # строки текущей части
    buf_len = 0
    fence_open = False       # открыт ли блок кода на конце текущей части

    def close_part() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunk = "".join(buf)
        if fence_open:
            chunk += CLOSE
        parts.append(chunk)
        # следующая часть начинается с переоткрытого блока, чтобы код читался целиком
        buf, buf_len = ([CLOSE] if fence_open else []), (len(CLOSE) if fence_open else 0)

    for line in text.splitlines(keepends=True):
        # место под закрывающий fence резервируем заранее: иначе часть выйдет за лимит
        room = limit - buf_len - (len(CLOSE) if fence_open else 0)
        while len(line) > max(room, 0):
            if room > 0:
                buf.append(line[:room])
                buf_len += room
                line = line[room:]
            close_part()
            room = limit - buf_len - (len(CLOSE) if fence_open else 0)
        buf.append(line)
        buf_len += len(line)
        if line.count(FENCE) % 2 == 1:
            fence_open = not fence_open

    if buf:
        parts.append("".join(buf))
    return [p for p in parts if p] or [text[:limit]]


def send_bot_reply(chat_id: int, text: str, reply_to_message_id: int = None):
    """
    Отдельная функция отправки: проверяет лимит символов Telegram (4096)
    и разбивает длинные сообщения, если ответ агента слишком большой.
    """
    parts = split_message(text)
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        kwargs = dict(
            # ссылки из веб-поиска раздувают сообщение превью-картинкой
            disable_web_page_preview=True,
            reply_to_message_id=reply_to_message_id if i == 0 else None,
        )
        try:
            bot.send_message(chat_id, part, parse_mode="Markdown", **kwargs)
        except telebot.apihelper.ApiTelegramException:
            # в заголовках и ссылках поиска встречаются _ и * — Markdown из них ломается,
            # поэтому этот кусок отправляем без разметки, а не теряем
            bot.send_message(chat_id, part, **kwargs)


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    """Подсказка по командам и возможным вопросам.

    Обработчик стоит ВЫШЕ catch-all: диспетчер telebot выбирает первый совпавший
    хендлер, а message_handler по умолчанию ловит только text — команда утонула бы в нём.
    """
    send_bot_reply(
        message.chat.id,
        "👋 Привет! Я отвечаю на вопросы про ИИ-агентов.\n\n"
        "Просто напиши вопрос текстом, например:\n"
        "• Чем долговременная память агента отличается от контекста одного шага?\n"
        "• Какие паттерны проектирования агентов есть в учебной базе?\n"
        "• Что нового вышло из LLM за последнюю неделю?\n\n"
        "По учебным материалам ищу в базе знаний (с указанием файла и страницы), "
        "по свежим событиям — в веб-поиске.\n/start — эта подсказка.",
    )


@bot.message_handler(content_types=["voice", "audio", "photo", "video", "document", "sticker"])
def handle_non_text(message):
    """Нетекст не доходит до агента — отвечаем сразу, иначе бот выглядит умершим."""
    bot.reply_to(
        message,
        "Я понимаю только текст 📝 Напиши вопрос словами — поиск по базе знаний "
        "и веб-поиск работают с текстовым запросом.",
    )


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Принимает сообщение от пользователя, вызывает агента и отправляет ответ."""
    try:
        question = (message.text or "").strip()
        if not question:
            bot.reply_to(message, "Я понимаю только текст 📝 Напиши вопрос словами.")
            return

        # Показываем статус "печатает...", пока агент думает
        bot.send_chat_action(message.chat.id, 'typing')

        # Получаем ответ от изолированной функции агента
        answer = ask_agent(question)

        # Отправляем ответ пользователю через отдельную функцию
        send_bot_reply(message.chat.id, answer, reply_to_message_id=message.message_id)

    except Exception as e:
        bot.reply_to(message, f"⚠️ Произошла ошибка: {str(e)}")


def warmup() -> None:
    """Поднимает Chroma и cross-encoder до первого сообщения.

    Без прогрева первый вопрос пользователю стоит десятки секунд на загрузку модели,
    а кривый .env или несобранный индекс выглядят как «бот молчит».
    """
    if os.getenv("BOT_WARMUP", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        print("⏭️ Прогрев отключён (BOT_WARMUP=0)")
        return
    try:
        started = time.time()
        get_retrieve()
        # реранкер внутри make_method поднимается лениво при первом retrieve(), а он стоит
        # несколько секунд — грузим заранее, чтобы не отдавать эту паузу первому вопросу
        if rag.RAG_USE_RERANKER:
            rag.get_reranker()
        print(f"🔥 Прогрев готов за {time.time() - started:.1f} с "
              f"(реранкер: {'вкл' if rag.RAG_USE_RERANKER else 'выкл'})")
    except Exception as e:
        print(f"⚠️ Прогрев не прошёл — поиск по базе знаний будет недоступен: "
              f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    warmup()
    print("🤖 Бот запущен и ожидает сообщений...")
    bot.infinity_polling()

    
