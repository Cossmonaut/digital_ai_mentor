import asyncio
import logging
import os
import re
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.enums import ParseMode
from aiogram.utils.text_decorations import html_decoration
from dotenv import load_dotenv

# Импортируем провайдеров
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.minimax import MiniMax
#from langchain_community.chat_models import GigaChat
from langchain_gigachat import GigaChat

# Наш класс RAG-системы
from graphrag import GraphRag

# Загружаем ключи
load_dotenv('../src/api_keys.env')

BOT_TOKEN = os.getenv("BOT_TOKEN")
CLOUD_RU_API_KEY = os.getenv("CLOUD_RU_API_KEY")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

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
    state_data = await state.get_data()
    raw_history = state_data.get("history", [])

    async def keep_typing():
        try:
            while True:
                await bot.send_chat_action(chat_id=message.chat.id, action="typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass

    typing_task = asyncio.create_task(keep_typing())
    reply = "Извините, не удалось сформировать ответ. Попробуйте перефразировать вопрос."

    try:
        # ИСПРАВЛЕНО: Если Claude думает/циклится дольше 25 секунд, мы принудительно обрываем её
        # и перенаправляем запрос в стабильный локальный GigaChat
        try:
            reply = await asyncio.wait_for(
                asyncio.to_thread(rag_system.get_response, user_query), 
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print("\n[Таймаут Claude]: Модель зависла в цикле. Переключаемся на GigaChat...")
            reply = await asyncio.to_thread(rag_system.gigachat_retriever, user_query)
        
        typing_task.cancel()
        await typing_task

        reply_html = convert_markdown_to_html(reply)
        await message.answer(reply_html, parse_mode=ParseMode.HTML)

        raw_history.append(("human", user_query))
        raw_history.append(("ai", reply))
        await state.update_data(history=raw_history[-20:]) 

    except Exception as e:
        logging.error(f"Ошибка при обработке сообщения: {e}")
        if not typing_task.done():
            typing_task.cancel()
        try:
            await message.answer(f"Произошла ошибка, отправляю сырой ответ:\n\n{reply}")
        except Exception:
            await message.answer("Извините, не удалось получить ответ от моделей. Попробуйте позже.")



# --- ЗАПУСК И ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ---
async def main():
    global rag_system

    print("Инициализация ИИ-моделей...")
    
    # 1. Возвращаем Cloud.ru, но переключаем модель на Claude
    main_llm = OpenAILike(
        api_base="https://foundation-models.api.cloud.ru/v1",
        api_key=CLOUD_RU_API_KEY,
        model="anthropic/claude-sonnet-4.6", 
        is_chat_model=True,
        temperature=0.3,
        timeout=60
    )

    # 2. Создаем MiniMax (резервный облачный провайдер)
    minimax_llm = MiniMax(
        model='MiniMax-M3',
        api_key=MINIMAX_API_KEY
    )

    # 3. Создаем бэкап-модель GigaChat через LangChain
    gigachat_llm = GigaChat(
        credentials=GIGACHAT_API_KEY,
        model="GigaChat:latest",
        verify_ssl_certs=False,
        scope='GIGACHAT_API_PERS'
    )

    print("Загрузка базы знаний и инициализация RAG пайплайна...")
    rag_system = GraphRag(llm=main_llm, gigachat_llm=gigachat_llm, minimax_llm=minimax_llm)

    print("Telegram-бот Ментор успешно запущен!")
    
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Бот остановлен.")
