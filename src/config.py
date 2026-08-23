"""
Централизованная конфигурация проекта.
Убирает жёсткие пути и дублирование os.getenv по всему коду.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Корень проекта (c:/Users/cossm/Desktop/digital_ mentor) ──
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Загрузка переменных окружения ──
ENV_PATH = BASE_DIR / "src" / "api_keys.env"
load_dotenv(ENV_PATH)

# ── API‑ключи ──
BOT_TOKEN = os.getenv("BOT_TOKEN")
CLOUD_RU_API_KEY = os.getenv("CLOUD_RU_API_KEY")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── Neo4j ──
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")

# ── Redis ──
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_TTL = int(os.getenv("REDIS_TTL", "86400"))  # 24 часа

# ── Пути ──
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "utils" / "prompts"
HF_MODELS_CACHE = BASE_DIR / "src" / "hf_models_cache"

# Локальная папка с моделью bge-m3 (если есть — офлайн‑режим)
LOCAL_MODEL_DIR = (
    HF_MODELS_CACHE
    / "models--BAAI--bge-m3"
    / "snapshots"
    / "5617a9f61b028005a4858fdac845db406aefb181"
)

# ── LLM‑параметры ──
MAIN_LLM_MODEL = os.getenv("MAIN_LLM_MODEL", "anthropic/claude-sonnet-4.6")
MAIN_LLM_BASE = os.getenv("MAIN_LLM_BASE", "https://foundation-models.api.cloud.ru/v1")
MAIN_LLM_TIMEOUT = int(os.getenv("MAIN_LLM_TIMEOUT", "60"))
MAIN_LLM_TEMPERATURE = float(os.getenv("MAIN_LLM_TEMPERATURE", "0.1"))

# ── Кеш ──
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 час

# ── Логирование ──
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "mentor.log"


def setup_logging() -> logging.Logger:
    """Настраивает логирование в файл + консоль с ротацией."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mentor")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Не дублируем хендлеры при повторном вызове
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Файл с ротацией (5 МБ, 3 архива)
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger
