import os
import yaml
import asyncio
import json
import nest_asyncio
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    PropertyGraphIndex,
    StorageContext,
    Settings
)

from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.minimax import MiniMax
from graph_knowledge_base import GraphKnowledgeBase

nest_asyncio.apply()
load_dotenv('../src/api_keys.env')
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
CLOUD_RU_API_KEY = os.getenv("CLOUD_RU_API_KEY")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")

NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = NEO4J_PASSWORD
NEO4J_URL = "bolt://localhost:7687"

# with open("graph_params.json", "r", encoding="utf-8") as f:
#     graph_params = json.load(f)


class GraphRag:
    def __init__(self, llm, gigachat_llm, minimax_llm):
        self.llm=llm
        self.gigachat_llm=gigachat_llm
        self.minimax_llm=minimax_llm
        self.graph_bd = GraphKnowledgeBase(self.llm)
        self.index = self.graph_bd.init_index()
        self.load_prompts()
        self.init_chat_engine()

    def load_prompts(self, 
                     prompt_path='../utils/prompts/prompts_rag.yaml',
                     gigachat_prompt_path='../utils/prompts/prompts_rag_gigachat.yaml'
                     ):
        with open(prompt_path, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.system_instruction = config.get("system_instruction")
        self.context_prompt = config.get('context_prompt')

        with open(gigachat_prompt_path, 'r', encoding='utf-8') as f:
            config_giga = yaml.safe_load(f)
        self.giga_system_instruction = config_giga.get("system_instruction")
        self.giga_context_prompt = config_giga.get('context_prompt')


    def init_chat_engine(self):
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="context",
            llm=self.llm,
            system_prompt=self.system_instruction,
            include_text=True,
            similarity_top_k=5,              
            context_prompt=self.context_prompt,
            verbose=False,
            timeout=30        
        )

    def get_response(self, query):
        try:
            response = self.chat_engine.chat(query)
            return response.response
        except:
            print('Не удалось воспользоваться llm api, перенаправляем ответ gigachat')
            return self.gigachat_retriever(query)

    def gigachat_retriever(self, query):
        retriever = self.index.as_retriever(
            similarity_top_k=5,
            include_text=True    
        )
        nodes_with_scores = retriever.retrieve(query)
        graph_relations = []
        text_chunks = []

        for node in nodes_with_scores:
            text_chunks.append(node.node.get_content())
            
            # Извлекаем имя сущности из метаданных чанка
            entity_name = node.node.metadata.get("entity_name") 
            
            if entity_name:
                try:
                    # ИСПРАВЛЕНО: Стучимся в graph_store через объект self.graph_bd
                    rel_triplets = self.graph_bd.graph_store.get_rel_map([entity_name], depth=1)
                    for source_node, rel_list in rel_triplets.items():
                        for rel in rel_list:
                            graph_relations.append(f"• {source_node} -> {rel.type} -> {rel.target_node}")
                except Exception:
                    pass

            if not graph_relations and "kg_rel_map" in node.node.metadata:
                for rel in node.node.metadata["kg_rel_map"]:
                    graph_relations.append(f"• {rel}")
        # Собираем уникальные строки контекста
        graph_str = "\n".join(set(graph_relations)) if graph_relations else "Прямых связей не обнаружено."
        text_str = "\n".join(set(text_chunks)) if text_chunks else "Текстовые выдержки отсутствуют."

        # БЕЗОПАСНОЕ форматирование промпта (передаем пустую историю для бэкапа)
        formatted_context = self.giga_context_prompt.format(
            graph_data=graph_str,
            text_data=text_str,
            chat_history="История недоступна (переключение на резервный канал).",
            query_str=query
        )

        # Собираем сообщения для LangChain GigaChat
        message_template = [
            SystemMessage(content=self.giga_system_instruction),
            HumanMessage(content=formatted_context)
        ]
        
        try:
            response = self.gigachat_llm.invoke(message_template)
            return response.content  # Возвращаем чистый текст ответа GigaChat
        except Exception as e:
            print(f"Ошибка самого GigaChat: {e}")
            return "К сожалению, не удалось обработать ваше сообщение. Попробуйте позже"
