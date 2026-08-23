import os
import nest_asyncio
from pathlib import Path

# Принудительно отключаем любые сетевые запросы Hugging Face ДО импорта библиотек ИИ
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from llama_index.core import (
    SimpleDirectoryReader,
    PropertyGraphIndex,
    StorageContext,
    Settings
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── Конфиг ──
from config import (
    NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_URL,
    LOCAL_MODEL_DIR, DATA_DIR, setup_logging,
)

nest_asyncio.apply()

logger = setup_logging()

class GraphKnowledgeBase:
    def __init__(self, llm):
        self.llm = llm
        self.graph_store = Neo4jPropertyGraphStore(
             username=NEO4J_USERNAME,
             password=NEO4J_PASSWORD,
             url=NEO4J_URL
        )
        self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        
        # ИСПРАВЛЕНО: Проверяем, существует ли указанная локальная папка с файлами модели
        try:
            print(f"[Офлайн]: Найдена локальная модель bge-m3 по пути: {LOCAL_MODEL_DIR}")
            # Передаем строковый путь К ПАПКЕ в качестве model_name. 
            # Это заставляет HuggingFaceEmbedding брать конфигурации прямо из папки без интернета.
            self.embed_model = HuggingFaceEmbedding(
                model_name=str(LOCAL_MODEL_DIR)
            )
        except:
            # Откат на стандартное поведение (например, для корпоративной сети, где интернет есть)
            print("[Сеть]: Локальная папка модели не найдена. Загрузка в кэш по умолчанию...")
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-m3",
                cache_folder="../utils/hf_models_cache"  
            )
        

    def init_index(self, update_flag=True, bd_path=None):
        bd_path = bd_path or str(DATA_DIR)
        new_docs = self.docs_graph_bd_update(bd_path)
        if new_docs and update_flag:
            logger.info(f"Обнаружены новые документы: {new_docs}")
            logger.info("Обновление базы данных...")
            # Передаем массив полных путей к новым файлам
            full_paths = [os.path.join(bd_path, doc) for doc in new_docs]
            return self.create_bd(full_paths)
        try:
            logger.info('Поиск существующей базы данных в Neo4j...')
            index = PropertyGraphIndex.from_existing(
                property_graph_store=self.graph_store,
                embed_model=self.embed_model,
                llm=self.llm
            )
        except Exception as e:
            logger.warning(f"База данных не найдена или пуста ({e}), инициализируем новую...")
            all_docs = [os.path.join(bd_path, f) for f in os.listdir(bd_path) if f.endswith('.txt')]
            index = self.create_bd(all_docs)
            
        return index

    def create_bd(self, document_list):
        documents = SimpleDirectoryReader(input_files=document_list).load_data()
        splitter = SentenceSplitter(chunk_size=384, chunk_overlap=40)
        nodes = splitter.get_nodes_from_documents(documents)
        index = PropertyGraphIndex(
            nodes=nodes,
            embed_model=self.embed_model,
            kg_extractors=[
                SchemaLLMPathExtractor(llm=self.llm)
                ],
            use_async=False,
            property_graph_store=self.graph_store,
            show_progress=True,
        )
        return index

    def docs_graph_bd_update(self, bd_path=None):
        bd_path = bd_path or str(DATA_DIR)
        result = self.graph_store.structured_query(
            "MATCH (n) WHERE n.file_name IS NOT NULL RETURN DISTINCT n.file_name AS file"
        )
        db_documents = set(doc['file'] for doc in result)
        if not os.path.exists(bd_path):
            return []
        current_dir_files = set(f for f in os.listdir(bd_path) if f.endswith('.txt'))
        new_files = current_dir_files.difference(db_documents)
        if not new_files:
            return []
            
        return list(new_files)
