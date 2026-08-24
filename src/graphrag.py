import os
import yaml
import asyncio
import json
import nest_asyncio
import re
import hashlib
import redis.asyncio as redis
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    PropertyGraphIndex,
    StorageContext,
    Settings
)

from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from graph_knowledge_base import GraphKnowledgeBase
from llama_index.core.indices.property_graph import PGRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.storage.chat_store.redis import RedisChatStore

from llm_factory import LLMFactory

# ── Конфиг ──
from config import (
    REDIS_URL, REDIS_TTL, CACHE_TTL_SECONDS,
    PROMPTS_DIR, setup_logging,
)

nest_asyncio.apply()

logger = setup_logging()


class GraphRag:
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        # Для индексации/достраивания графа используем default-модель.
        self.graph_bd = GraphKnowledgeBase(self.llm_factory.get_default_llm())
        self.index = self.graph_bd.init_index()
        self.load_prompts()

        # ── Ретривер и постпроцессор (shared, создаются один раз) ──
        self.custom_retriever = self.index.as_retriever(
            similarity_top_k=7,
            include_text=True
        )
        # ИСПРАВЛЕНО: теперь self.score_filter, а не локальная переменная
        self.score_filter = SimilarityPostprocessor(similarity_cutoff=0.75)

        # ── Redis: клиент для кеша ответов ──
        self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)

        # ── Redis: chat store для per-user истории ──
        try:
            self.chat_store = RedisChatStore(
                redis_url=REDIS_URL,
                ttl=REDIS_TTL
            )
            logger.info("[Redis-Лог]: Центральное хранилище истории диалогов успешно подключено! 💾")
        except Exception as e:
            logger.error(f"[Redis-Ошибка]: Не удалось подключить Redis ({e}). Откат на In-Memory.")
            self.chat_store = None

        # Кеш chat‑движков per-user и per-model (создание ContextChatEngine дешёвое — держим в памяти)
        self._user_engines: dict[tuple[str, str], ContextChatEngine] = {}

    def load_prompts(self,
                     prompt_path=None,
                     gigachat_prompt_path=None
                     ):
        prompt_path = prompt_path or str(PROMPTS_DIR / "prompts_rag.yaml")
        gigachat_prompt_path = gigachat_prompt_path or str(PROMPTS_DIR / "prompts_rag_gigachat.yaml")

        with open(prompt_path, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.system_instruction = config.get("system_instruction")
        self.context_prompt = config.get('context_prompt')

        with open(gigachat_prompt_path, 'r', encoding='utf-8') as f:
            config_giga = yaml.safe_load(f)
        self.giga_system_instruction = config_giga.get("system_instruction")
        self.giga_context_prompt = config_giga.get('context_prompt')

    def _create_chat_engine(self, llm, user_id: str) -> ContextChatEngine:
        """Создаёт ContextChatEngine с per-user Redis‑backed памятью."""
        if self.chat_store:
            memory = ChatMemoryBuffer.from_defaults(
                chat_store=self.chat_store,
                chat_store_key=f"history:{user_id}",
                token_limit=3000
            )
        else:
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        return ContextChatEngine.from_defaults(
            llm=llm,
            system_prompt=self.system_instruction,
            context_prompt=self.context_prompt,
            verbose=False,
            timeout=60,
            retriever=self.custom_retriever,
            node_postprocessor=[self.score_filter],
            memory=memory
        )

    def get_user_chat_engine(self, llm, model_id: str, user_id: str) -> ContextChatEngine:
        """Возвращает (или создаёт) chat engine для конкретного пользователя и модели."""
        key = (user_id, model_id)
        if key not in self._user_engines:
            self._user_engines[key] = self._create_chat_engine(llm, user_id)
        return self._user_engines[key]

    async def get_response_async(self, query: str, history=None, user_id: str = None) -> str:
        """Каскад с health-check: default → fallback с кешированием в Redis."""

        # ── 1. Проверяем кеш ответа ──
        cache_key = None
        if user_id:
            cache_key = f"cache:answer:{user_id}:{hashlib.md5(query.encode()).hexdigest()}"
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    logger.info(f"[Cache HIT] user={user_id} query_hash={cache_key[-12:]}")
                    return cached
                logger.info(f"[Cache MISS] user={user_id} query_hash={cache_key[-12:]}")
            except Exception as e:
                logger.warning(f"[Cache]: Не удалось проверить кеш ({e}). Продолжаем без кеша.")

        # ── 2. Выбираем первую доступную модель через фабрику ──
        try:
            resolved = await self.llm_factory.get_available_llm()
            logger.info(
                f"[RAG-Лог]: Запрос направлен в модель {resolved.model_id} "
                f"(провайдер {resolved.provider}), user={user_id}"
            )
        except Exception as e:
            logger.error(f"[RAG-Крах]: Ни одна модель не доступна: {e}", exc_info=True)
            return "Извините, система временно перегружена. Пожалуйста, повторите запрос позже."

        # ── 3. Основной контур (OpenAI-like / MiniMax) ──
        if resolved.provider != "gigachat":
            chat_engine = (
                self.get_user_chat_engine(resolved.llm, resolved.model_id, user_id)
                if user_id
                else self._create_chat_engine(resolved.llm, "default")
            )
            try:
                response = await asyncio.wait_for(chat_engine.achat(query), timeout=45.0)
                if response and response.response:
                    # Кешируем ответ
                    if cache_key:
                        try:
                            await self.redis_client.setex(
                                cache_key, CACHE_TTL_SECONDS, response.response
                            )
                            logger.info(f"[Cache WRITE] user={user_id} ttl={CACHE_TTL_SECONDS}s")
                        except Exception as e:
                            logger.warning(f"[Cache]: Не удалось записать в кеш ({e}).")
                    return response.response
                raise ValueError(f"Пустой ответ от модели {resolved.model_id}")
            except Exception as e:
                logger.warning(
                    f"[RAG-Предупреждение]: Модель {resolved.model_id} дала сбой или таймаут ({e}). "
                    f"Переход на следующую доступную модель..."
                )
                # Пробуем fallback, исключая текущую модель
                return await self._try_fallback(
                    query, cache_key, exclude_model_id=resolved.model_id, user_id=user_id
                )

        # ── 4. Резервный контур GigaChat ──
        logger.info(f"[RAG-Лог]: Запуск резервного сценария GigaChat... user={user_id}")
        try:
            response = await self.gigachat_retriever_async(
                query, llm=resolved.llm, user_id=user_id
            )
            if cache_key and response:
                try:
                    await self.redis_client.setex(cache_key, CACHE_TTL_SECONDS, response)
                except Exception:
                    pass
            return response
        except Exception as e:
            logger.error(f"[RAG-Крах]: Сбой GigaChat: {e}", exc_info=True)
            return await self._try_fallback(
                query, cache_key, exclude_model_id=resolved.model_id, user_id=user_id
            )

    async def _try_fallback(
        self,
        query: str,
        cache_key: str | None,
        exclude_model_id: str,
        user_id: str | None,
    ) -> str:
        """Пробует следующую доступную модель, исключая уже отработавшую."""
        try:
            resolved = await self.llm_factory.get_available_llm(exclude=[exclude_model_id])
            logger.info(f"[RAG-Лог]: Fallback в модель {resolved.model_id}, user={user_id}")
        except Exception as e:
            logger.error(f"[RAG-Крах]: Fallback невозможен: {e}", exc_info=True)
            return "Извините, система временно перегружена. Пожалуйста, повторите запрос позже."

        if resolved.provider == "gigachat":
            response = await self.gigachat_retriever_async(
                query, llm=resolved.llm, user_id=user_id
            )
        else:
            chat_engine = (
                self.get_user_chat_engine(resolved.llm, resolved.model_id, user_id)
                if user_id
                else self._create_chat_engine(resolved.llm, "default")
            )
            response_obj = await asyncio.wait_for(chat_engine.achat(query), timeout=45.0)
            response = response_obj.response if response_obj and response_obj.response else ""

        if cache_key and response:
            try:
                await self.redis_client.setex(cache_key, CACHE_TTL_SECONDS, response)
            except Exception:
                pass
        return response

    async def gigachat_retriever_async(self, query: str, llm, user_id: str = None) -> str:
        """Асинхронный отказоустойчивый ретривер для GigaChat."""
        # Извлекаем ноды напрямую через кастомный ретривер
        nodes_with_scores = await self.custom_retriever.aretrieve(query)

        # Применяем постпроцессор весов (теперь self.score_filter существует)
        try:
            nodes_with_scores = self.score_filter.postprocess_nodes(nodes_with_scores)
        except Exception:
            pass

        graph_relations = []
        text_chunks = []
        entity_names = []

        for node in nodes_with_scores:
            text_chunks.append(node.node.get_content())
            entity_name = node.node.metadata.get("entity_name")
            if entity_name:
                entity_names.append(entity_name)

        if entity_names:
            try:
                rel_triplets = await asyncio.to_thread(
                    self.graph_bd.graph_store.get_rel_map, entity_names, depth=1
                )
                for source_node, rel_list in rel_triplets.items():
                    for rel in rel_list:
                        graph_relations.append(f"• {source_node} -> {rel.type} -> {rel.target_node}")
            except Exception as e:
                logger.error(f"[Neo4j-Ошибка]: {e}")

        # Страховочный добор связей
        if not graph_relations:
            for node in nodes_with_scores:
                if "kg_rel_map" in node.node.metadata:
                    for rel in node.node.metadata["kg_rel_map"]:
                        graph_relations.append(f"• {rel}")

        # ── Загружаем историю диалога из Redis (если есть) ──
        chat_history_str = "История недоступна (резервный канал)."
        if user_id and self.chat_store:
            try:
                history_items = await self.chat_store.aget_messages(f"history:{user_id}")
                if history_items:
                    chat_history_str = "\n".join(
                        f"{'👤' if m.role.value == 'user' else '🤖'}: {m.content}"
                        for m in history_items
                    )
            except Exception as e:
                logger.warning(f"[Redis]: Не удалось загрузить историю для GigaChat ({e}).")

        # Сборка финальных строк контекста
        graph_str = "\n".join(set(graph_relations)) if graph_relations else "Прямых связей в графе не найдено."
        text_str = "\n".join(set(text_chunks)) if text_chunks else "В базе знаний нет подходящих текстовых выдержек."

        logger.info(f"[DEBUG]: До GigaChat дошло {len(text_chunks)} текстовых чанков. user={user_id}")

        formatted_context = self.giga_context_prompt.format(
            graph_data=graph_str,
            text_data=text_str,
            chat_history=chat_history_str,
            query_str=query
        )

        message_template = [
            SystemMessage(content=self.giga_system_instruction),
            HumanMessage(content=formatted_context)
        ]

        # Вызов GigaChat через асинхронный метод langchain ainvoke
        response = await llm.ainvoke(message_template)
        return response.content

    def compact_context(self):
        pass
