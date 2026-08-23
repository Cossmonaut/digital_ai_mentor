import asyncio
import logging
import os
import re
import aiohttp
import redis
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.enums import ParseMode
from aiogram.utils.text_decorations import html_decoration
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.utils.chat_action import ChatActionSender

# Импортируем провайдеров
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.minimax import MiniMax
from langchain_gigachat import GigaChat

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Наш класс RAG-системы
from graphrag import GraphRag

# ── Конфиг ──
from config import (
    BOT_TOKEN, CLOUD_RU_API_KEY, MINIMAX_API_KEY, GIGACHAT_API_KEY,
    MAIN_LLM_MODEL, MAIN_LLM_BASE, MAIN_LLM_TIMEOUT, MAIN_LLM_TEMPERATURE,
    setup_logging,
)

# Адрес вашего домашнего прокси из скриншота.
# Если прокси типа SOCKS5, замените "http://" на "socks5://"
PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:1443")

# Инициализируем глобальные переменные (сам bot создадим внутри main после проверки сети)
bot = None
dp = Dispatcher()
logger = setup_logging()

rag_system = None

def convert_markdown_to_html(text: str) -> str:
    """Конвертирует базовый Markdown от ИИ в безопасный HTML для Telegram."""
    if not text:
        return ""
        
    # ИСПРАВЛЕНО: Удаляем некрасивые маркдаун-разделители "---" 
    # и заменяем их на пустую строку с переносом, чтобы визуально разделить блоки
    text = re.sub(r'(?m)^[ \t]*---[ \t]*$', '\n', text)
    
    # Стандартное экранирование жирного шрифта
    parts = text.split("**")
    for i in range(len(parts)):
        parts[i] = html_decoration.quote(parts[i])
    for i in range(1, len(parts), 2):
        parts[i] = f"<b>{parts[i]}</b>"
    text = "".join(parts)
    
    # Перевод списков в буллиты
    text = re.sub(r'(?m)^[ \t]*[-\*][ \t]+', '• ', text)
    
    return text


# --- ХЕНДЛЕРЫ TELEGRAM ---
@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()  # Полностью очищаем старую историю диалога
    start_text = "Привет! Я твой <b>ИИ-ментор</b>. С какими профессиональными вызовами или трудностями ты столкнулся сегодня? Расскажи подробнее, и мы разберем это."
    await message.answer(start_text, parse_mode=ParseMode.HTML)




@dp.message(F.text)
async def handle_user_message(message: Message, state: FSMContext):
    user_query = message.text
    
    # 1. Получаем историю чата (с защитой от None)
    state_data = await state.get_data() or {}
    raw_history = state_data.get("history", [])

    # Инициализируем дефолтный ответ на случай непредвиденных сбоев
    reply = "Извините, не удалось сформировать ответ. Попробуйте перефразировать вопрос."

    # 2. Менеджер контекста aiogram автоматически крутит анимацию "typing" в фоне.
    # Больше не нужны фоновые задачи (asyncio.create_task), костыли с бесконечным циклом и их ручная отмена.
    async with ChatActionSender.typing(chat_id=message.chat.id, bot=message.bot):
        try:
            # Вызываем единый асинхронный метод RAG-системы.
            # Задаем жесткий общий таймаут на выполнение всей цепочки переключений (например, 35 секунд)
            reply = await asyncio.wait_for(
                rag_system.get_response_async(
                    user_query, start_llm='claude', history=raw_history,
                    user_id=str(message.from_user.id)
                ),
                timeout=90.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Общий таймаут RAG-системы превышен для пользователя {message.from_user.id}")
            reply = "К сожалению, система перегружена и не успела ответить вовремя. Пожалуйста, повторите запрос."
        except Exception as e:
            logger.error(f"Критическая ошибка RAG-системы: {e}", exc_info=True)
            reply = "Произошла внутренняя ошибка при обработке запроса. Мы уже чиним её!"

    # 3. Отправка ответа пользователю и обновление контекста
    try:
        # Проверяем, что ответ вообще сформирован (не None и не пустой)
        if not reply:
            reply = "Извините, не удалось получить вменяемый ответ от нейросетей."

        reply_html = convert_markdown_to_html(reply)
        await message.answer(reply_html, parse_mode=ParseMode.HTML)

        # Сохраняем в историю только если запрос прошел успешно
        raw_history.append(("human", user_query))
        raw_history.append(("ai", reply))
        
        # Храним последние 20 сообщений (10 диалогов), срезая старые
        await state.update_data(history=raw_history[-20:]) 

    except Exception as e:
        logger.error(f"Ошибка при отправке сообщения или сохранении FSM: {e}", exc_info=True)
        # Страховочный ответ без разметки HTML, если convert_markdown_to_html выдал кривой тег
        try:
            await message.answer("Произошла ошибка при отображении ответа. Попробуйте отправить запрос еще раз.")
        except Exception:
            pass



async def check_direct_connection() -> bool:
    """Проверяет прямой доступ к API Telegram (для корпоративной сети)."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://telegram.org", timeout=3.0) as resp:
                return resp.status in [200, 404]  # Любой ответ от сервера означает, что блокa нет
    except Exception:
        return False


# --- ЗАПУСК И ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ---
async def main():
    global rag_system, bot

    print("[Сеть]: Проверка доступности Telegram...")
    direct_connect = await check_direct_connection()

    if direct_connect:
        print("[Сеть]: Прямое подключение доступно (Корпоративная сеть). Запуск без прокси.")
        bot = Bot(token=BOT_TOKEN)
    else:
        print(f"[Сеть]: Прямое подключение отсутствует. Настройка домашнего прокси {PROXY_URL}...")
        # Применяем прокси для Telegram-бота через сессию
        session = AiohttpSession(proxy=PROXY_URL)
        bot = Bot(token=BOT_TOKEN, session=session)
        
        # Передаем прокси в переменные окружения для ИИ-моделей (LlamaIndex / LangChain)
        os.environ["HTTP_PROXY"] = PROXY_URL
        os.environ["HTTPS_PROXY"] = PROXY_URL

    logger.info("Инициализация ИИ-моделей...")

    # 1. Возвращаем Cloud.ru, но переключаем модель на Claude
    main_llm = OpenAILike(
        api_base=MAIN_LLM_BASE,
        api_key=CLOUD_RU_API_KEY,
        model=MAIN_LLM_MODEL,
        is_chat_model=True,
        temperature=MAIN_LLM_TEMPERATURE,
        timeout=MAIN_LLM_TIMEOUT
    )

    # 2. Создаем MiniMax (резервный облачный провайдер)
    minimax_llm = MiniMax(
        model='MiniMax-M3',
        api_key=MINIMAX_API_KEY,
        temperature=MAIN_LLM_TEMPERATURE
    )

    # 3. Создаем бэкап-модель GigaChat через LangChain
    gigachat_llm = GigaChat(
        credentials=GIGACHAT_API_KEY,
        model="GigaChat:latest",
        verify_ssl_certs=False,
        scope='GIGACHAT_API_PERS',
        temperature=MAIN_LLM_TEMPERATURE
    )

    logger.info("Загрузка базы знаний и инициализация RAG пайплайна...")
    rag_system = GraphRag(llm=main_llm, gigachat_llm=gigachat_llm, minimax_llm=minimax_llm)

    logger.info("Telegram-бот Ментор успешно запущен!")

    await bot.delete_webhook(drop_pending_updates=True)
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
        logger.info("Бот остановлен, сессия закрыта.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
