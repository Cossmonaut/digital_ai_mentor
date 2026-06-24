import asyncio
import logging
import os
import re  # Добавили для обработки списков нейросети
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.enums import ParseMode
from aiogram.utils.text_decorations import html_decoration  # Официальный инструмент экранирования
from dotenv import load_dotenv

# Импортируем типы сообщений LangChain, которые ожидает lcel_chain
from langchain_core.messages import HumanMessage, AIMessage 
from llm import LLMApiModule

# Загружаем ключи из .env
load_dotenv('api_keys.env')

BOT_TOKEN = os.getenv("BOT_TOKEN")
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")

bot = Bot(token=BOT_TOKEN)
llm_api = LLMApiModule(api_key=GIGACHAT_API_KEY)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)


def convert_markdown_to_html(text: str) -> str:
    """
    Конвертирует базовый Markdown от GigaChat в валидный и безопасный HTML для Telegram.
    Решает проблему падений из-за спецсимволов.
    """
    # 1. Защищаем текст: экранируем <, >, & через стандартный метод aiogram
    # Но временно сохраняем маркдаун-звездочки, чтобы не сломать их
    parts = text.split("**")
    for i in range(len(parts)):
        parts[i] = html_decoration.quote(parts[i])
    
    # Собираем текст обратно, оборачивая каждую нечетную часть в <b>...</b>
    for i in range(1, len(parts), 2):
        parts[i] = f"<b>{parts[i]}</b>"
    text = "".join(parts)
    
    # 2. Переводим списки из Markdown (- пункт или * пункт) в аккуратные списки Telegram
    # Заменяем дефис/звездочку в начале строки на красивую точку-буллит (•)
    text = re.sub(r'(?m)^[ \t]*[-\*][ \t]+', '• ', text)
    
    return text


# --- ХЕНДЛЕРЫ TELEGRAM ---
@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    # При старте очищаем старую историю диалога
    await state.clear()
    
    # В HTML режиме жирный текст пишется через тег <b>
    start_text = "Привет! Я твой <b>ИИ-ментор</b>. С какими профессиональными вызовами или трудностями ты столкнулся сегодня? Расскажи подробнее, и мы разберем это."
    
    await message.answer(start_text, parse_mode=ParseMode.HTML)


@dp.message(F.text)
async def handle_user_message(message: Message, state: FSMContext):
    user_query = message.text

    # Показываем статус "печать текста", пока ИИ генерирует ответ
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    # 1. Получаем историю из FSM-контекста aiogram
    state_data = await state.get_data()
    raw_history = state_data.get("history", [])

    # 2. Конвертируем сохраненную историю в формат объектов LangChain
    chat_history = []
    for role, text in raw_history:
        if role == "human":
            chat_history.append(HumanMessage(content=text))
        elif role == "ai":
            chat_history.append(AIMessage(content=text))

    try:
        # 3. Передаем запрос вместе с объектами истории в ваш LLM-модуль
        reply = await llm_api.get_response(user_query, history=chat_history)
        
        # 4. Преобразуем Markdown-ответ от GigaChat в безопасный HTML
        reply_html = convert_markdown_to_html(reply)
        
        # Отправляем сообщение в режиме HTML
        await message.answer(reply_html, parse_mode=ParseMode.HTML)

        # 5. Сохраняем оригинальные реплики в историю (не более 10 последних шагов)
        raw_history.append(("human", user_query))
        raw_history.append(("ai", reply))
        await state.update_data(history=raw_history[-20:]) 

    except Exception as e:
        logging.error(f"Ошибка при обработке сообщения: {e}")
        # Защитный механизм: если HTML всё равно сломался, отправляем как обычный текст без разметки
        try:
            await message.answer(reply)
        except Exception:
            await message.answer("Произошла ошибка при отправке ответа. Попробуйте позже.")


# --- ЗАПУСК БОТА ---
async def main():
    print("Бот-ментор запущен и готов к работе...")
    # Очищаем очередь старых сообщений при запуске, чтобы бот не спамил ответами
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
