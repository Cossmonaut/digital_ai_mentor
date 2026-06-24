import yaml
import sys
import asyncio
from knowledge_base import KnowledgeBase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_gigachat.chat_models import GigaChat

sys.path.append('..')

class LLMApiModule:
    def __init__(self, api_key, model="GigaChat:latest"):
        # Официальный клиент GigaChat из правильного пакета langchain_gigachat
        self.ai_client = GigaChat(
            credentials=api_key,
            model=model,
            verify_ssl_certs=False,
            scope='GIGACHAT_API_PERS'
        )
        
        # Загружаем системные инструкции из yaml-файла
        with open(r"../utils/prompts.yaml", "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

        # Строим шаблон промпта с поддержкой истории диалога и контекста методички
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.prompts['system_prompt'] + "\n\nИспользуй этот контекст методички для ответа:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Инициализируем нашу стабильную локальную базу знаний
        self.kn_base = KnowledgeBase()
        self.kn_base.init_base() # Сразу индексируем тексты в оперативную память
        
        # Цепочка LCEL: принимает готовые данные, подставляет в промпт и отдает модели
        self.lcel_chain = (
            self.prompt_template
            | self.ai_client  
            | StrOutputParser()
        )

    async def get_response(self, user_request, history=None):
        chat_history = history if history is not None else []

        # 1. Запускаем поиск по методичке в отдельном потоке (asyncio.to_thread).
        # Это предотвращает зависание Телеграм-бота во время тяжелых математических расчетов векторов.
        context = await asyncio.to_thread(self.kn_base.search, user_request, k=3)

        # 2. Асинхронно отправляем сформированный промпт в GigaChat по сети
        response = await self.lcel_chain.ainvoke({
            "context": context,
            "input": user_request,
            "chat_history": chat_history
        })
        
        # Возвращаем чистую строку ответа, так как StrOutputParser() уже убрал метаданные
        return response
