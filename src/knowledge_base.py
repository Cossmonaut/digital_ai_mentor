import numpy as np
from sentence_transformers import SentenceTransformer, util

class KnowledgeBase:
    def __init__(self, api_key=None):
        # Официальный класс напрямую от HuggingFace (без посредников LangChain)
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.chunks = []
        self.chunk_embeddings = None

    def get_chunks(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as t:
            txt_file = t.read()
        return [c.strip() for c in txt_file.split('---CHUNK_SPLIT---') if c.strip()]

    def init_base(self, txt_path='../data/mentor_basics_1.txt'):
        """Этот метод один раз прочитает текст и сделает из него векторы"""
        self.chunks = self.get_chunks(txt_path)
        # Генерируем матрицу векторов для всех чанков методички
        self.chunk_embeddings = self.model.encode(self.chunks, convert_to_tensor=True)

    def search(self, query, k=3):
        """Обычный семантический поиск по тексту методички"""
        if not self.chunks:
            self.init_base()
            
        # Кодируем вопрос пользователя в вектор
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Считаем схожесть вопроса со всеми чанками
        cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
        
        # Берем топ-K самых похожих
        top_results = np.argpartition(-cos_scores.cpu(), k)[:k]
        
        # Собираем найденный текст
        retrieved_chunks = [self.chunks[idx] for idx in top_results]
        return "\n\n".join(retrieved_chunks)
