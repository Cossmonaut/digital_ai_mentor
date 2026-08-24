# AGENTS.md — Digital AI Mentor

Этот файл предназначен для AI-агентов, работающих с кодовой базой. Читатель ничего не знает о проекте заранее.

## Обзор проекта

**Digital Mentor** — Telegram-бот корпоративного ИИ-ментора, построенный на GraphRAG: граф знаний в Neo4j (через LlamaIndex `PropertyGraphIndex`) + каскад LLM-провайдеров с автоматическим фолбэком. Бот отвечает от лица персоны ментора (см. системные промпты в `utils/prompts/`), опираясь на базу знаний из текстовых файлов (`data/`) — сейчас это Кодекс этики (`ethic_code.txt`) и руководство по менторингу (`mentor_basics_1.txt`).

Архитектура runtime:

```
Telegram Bot (aiogram 3)  ──▶  GraphRAG Engine (LlamaIndex)  ──▶  Neo4j (граф знаний)
                                    │        ▲
                                    ▼        │
                              Redis (кэш ответов +      LLM-каскад: фабрика провайдеров
                              история диалогов)         с health-check: default (Cloud.ru)
                                                      → fallback-модели из config/models.json
```

- Язык: **Python 3.11** (образ `python:3.11-slim`).
- Все комментарии, логи, промпты и документация в проекте — **на русском языке**. Новый код и комментарии тоже следует писать на русском.
- MiniMax (`llama-index-llms-minimax`, модель `MiniMax-M3`) описан в `config/models.json` и участвует в fallback-цепочке после GigaChat. Порядок моделей управляется фабрикой `src/llm_factory.py`.
- `OPENROUTER_API_KEY` читается в `config.py`, но нигде в коде не используется.

## Структура кода и модули

```
docker-compose.yml           # Оркестрация в корне: app + neo4j:5.26.0 + redis:7-alpine,
                             # все параметры подставляются из .env (порты, NEO4J_AUTH и т.д.)
.env                         # Секреты и все параметры запуска (НЕ коммитить; создаётся из .env.example)
start.sh                     # Запуск стека; если app уже запущен — только выводит статус и выходит
stop.sh                      # Остановка стека (docker compose down), данные в ./docker/ сохраняются
update.sh                    # Обновление: pull базовых образов, пересборка app, перезапуск
entrypoint.sh                # Entrypoint контейнера app: проверка env, ожидание Neo4j/Redis, запуск бота

config/
└── models.json              # Описание default-модели, fallback_order и параметров LLM-провайдеров
                             # (base_url, api_key_env, model_id, temperature, timeout и т.д.)

src/
├── main_rag.py              # Точка входа: Telegram-бот (aiogram 3), хендлеры /start и F.text,
│                            # проверка сети/прокси, создание LLMFactory, запуск polling.
│                            # Запускается как модуль: python -m src.main_rag
├── graphrag.py              # Класс GraphRag: получает LLM через фабрику, Redis-кэш ответов,
│                            # per-user/per-model ContextChatEngine с RedisChatStore-историей,
│                            # резервный ретривер gigachat_retriever_async (ручная сборка контекста).
├── graph_knowledge_base.py  # Класс GraphKnowledgeBase: Neo4jPropertyGraphStore, эмбеддинги
│                            # bge-m3 (HuggingFaceEmbedding, офлайн из локального кэша),
│                            # создание/обновление PropertyGraphIndex через
│                            # SchemaLLMPathExtractor + SentenceSplitter (chunk 384 / overlap 40).
├── llm_factory.py           # Фабрика LLM-провайдеров: загружает config/models.json,
│                            # проверяет доступность default (Cloud.ru) через GET /models,
│                            # при недоступности перебирает fallback_order и возвращает рабочий LLM.
└── config.py                # Централизованная конфигурация: загрузка .env из корня через
                             # python-dotenv (фолбэк: src/api_keys.env — legacy), все os.getenv
                             # собраны здесь, setup_logging()
                             # (RotatingFileHandler logs/mentor.log, 5 МБ × 3 архива, + консоль).

utils/
├── parse_pdf.py             # Офлайн-утилита парсинга PDF в .txt (PyMuPDF/fitz), чанкование
├── parse_yt_txt.ipynb       # Ноутбук парсинга YouTube-транскриптов
└── prompts/
    ├── prompts_rag.yaml           # Промпты основной LLM (system_instruction + context_prompt)
    └── prompts_rag_gigachat.yaml  # Промпты резервной LLM GigaChat

data/                        # .txt файлы базы знаний (монтируются read-only в /app/data)
docker/                      # Данные контейнеров: neo4j/{data,plugins,logs}, redis/data, logs
                             # (bind-монты compose; создаются start.sh/update.sh, в .gitignore)
logs/                        # Логи при локальном запуске вне Docker (в Docker логи пишутся в docker/logs)
```

Важные детали реализации:

- Импорты в `src/` — плоские (`from graphrag import GraphRag`, `from config import ...`), не пакетные: код рассчитан на запуск с `PYTHONPATH=/app` и командой `python -m src.main_rag` (так делает `entrypoint.sh` в Docker).
- Офлайн-режим Hugging Face: `src/graph_knowledge_base.py:6-7` выставляет `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` до импорта ИИ-библиотек, `src/main_rag.py:21-23` — `HF_DATASETS_OFFLINE`/`TRANSFORMERS_OFFLINE`. Эмбеддинг-модель bge-m3 загружается из локального кэша `src/hf_models_cache/models--BAAI--bge-m3/snapshots/<hash>/` (путь зашит в `config.py` как `LOCAL_MODEL_DIR`). Если локальной папки нет — откат на скачивание `BAAI/bge-m3` в `utils/hf_models_cache`.
- Индексация: при старте `GraphKnowledgeBase.init_index()` сравнивает имена `.txt` в `data/` со значениями `file_name` в Neo4j и доиндексирует только новые файлы; если граф пуст — индексирует все файлы. Полная переиндексация — очистка графа (`MATCH (n) DETACH DELETE n`) и перезапуск.
- Кэш ответов в Redis: ключ `cache:answer:{user_id}:{md5(query)}`, TTL = `CACHE_TTL_SECONDS` (1 ч). История диалогов: `RedisChatStore` с ключом `history:{user_id}`, TTL = `REDIS_TTL` (24 ч); chat-движки кэшируются per-user **и per-model** в памяти (`self._user_engines`), потому что разные fallback-модели требуют разных `ContextChatEngine`.
- Таймауты: весь RAG-вызов — 90 с (`asyncio.wait_for` в `handle_user_message`, `src/main_rag.py`), основной LLM (Claude) — 45 с (`src/graphrag.py`), дальше fallback на следующую доступную модель. HTTP-таймауты и температура задаются в `config/models.json` для каждой модели.
- История диалога дублируется: помимо RedisChatStore, хендлер хранит последние 20 сообщений в FSM-состоянии aiogram (`state.update_data(history=...)`).
- GigaChat вызывается через LangChain (`langchain_gigachat.GigaChat.ainvoke`) с ручной сборкой контекста: текстовые чанки из ретривера + связи графа (`get_rel_map`, depth=1) + история из Redis.
- Fallback теперь централизован: `LLMFactory.get_available_llm()` проверяет `default` модель через GET `{base_url}/models` с токеном из env; если проверка не проходит, перебирает модели из `fallback_order` по их `model_id`. Возвращается объект `ResolvedModel` с полями `llm`, `model_id`, `provider`. `GraphRag` выбирает тип chat engine по `provider`, а не по имени модели, поэтому у одного провайдера может быть сколько угодно моделей в `fallback_order`.

## Конфигурация

- `.env` в корне проекта — единственный источник секретов (шаблон: `.env.example`). Обязательные переменные (проверяются в `entrypoint.sh`): `BOT_TOKEN`, `CLOUD_RU_API_KEY`, `GIGACHAT_API_KEY`, `MINIMAX_API_KEY`, `NEO4J_PASSWORD`. `src/api_keys.env` — legacy-фолбэк, который `config.py` подхватывает, если корневого `.env` нет.
- `config/models.json` — declarative конфиг LLM-моделей: `default` и `fallback_order` содержат **идентификаторы конкретных моделей** из секции `models`. Каждая модель описывается полями `provider`, `base_url`, `api_key_env`, `model_id`, `temperature`, `timeout` и специфичными параметрами (например, `scope`/`verify_ssl_certs` для GigaChat). У одного провайдера может быть несколько моделей с разными `model_id`. API-ключи не хранятся в JSON — там только имена переменных окружения.
- Все настраиваемые параметры с дефолтами собраны в `src/config.py` — **не добавляйте `os.getenv` в других файлах**, расширяйте `config.py` (исключение из legacy: `PROXY_URL` читается напрямую в `src/main_rag.py`).
- Ключевые переменные: `NEO4J_URL` (по умолч. `bolt://localhost:7687`, в Docker `bolt://neo4j:7687`), `REDIS_URL` (в Docker `redis://redis_cache:6379`), `CACHE_TTL_SECONDS`, `REDIS_TTL`, `LOG_LEVEL`, `PROXY_URL` (опционально, для доступа к Telegram API через прокси; по умолчанию `http://127.0.0.1:1443`). Параметры LLM (`base_url`, `model_id`, `timeout`, `temperature`) теперь живут в `config/models.json`.
- Прокси: при старте бот проверяет прямой доступ к `https://telegram.org`; если его нет — Telegram-сессия и `HTTP_PROXY`/`HTTPS_PROXY` для LLM-клиентов настраиваются через `PROXY_URL`.

## Сборка и запуск

Проект **не имеет** `pyproject.toml`/`setup.py` — зависимости только через pinned `requirements.txt`. Локальный запуск вне Docker не является основным сценарием; основной путь — Docker Compose из корня проекта:

```bash
# Однократная подготовка
cp .env.example .env   # заполнить ключи

# Запуск / остановка / обновление (скрипты в корне)
./start.sh     # если app уже запущен — выведет статус и выйдет, дубликат не поднимет
./stop.sh      # docker compose down, данные в ./docker/ сохраняются
./update.sh    # pull базовых образов + пересборка app + перезапуск

# Логи / статус
docker compose logs -f app
docker compose ps
```

Внешние named volumes больше не нужны: данные Neo4j и Redis и логи приложения лежат в bind-монтах `./docker/` (в `.gitignore`).

Dockerfile: один образ `python:3.11-slim` (multi-stage не используется), `pip install -r requirements.txt`, запуск от непривилегированного пользователя `appuser` через `entrypoint.sh`, который проверяет обязательные env-переменные, ждёт готовности Neo4j (bolt:7687) и Redis (6379) через `nc` и затем выполняет `exec python -m src.main_rag`.

Полезно знать: healthcheck в Dockerfile проверяет `curl http://localhost:8000/health`, но HTTP-сервера в приложении **нет** (бот работает через long polling) — этот healthcheck фактически всегда падает; compose переопределяет его на тривиальный `python -c "import sys; sys.exit(0)"`.

Доступ к сервисам с хоста: Neo4j Browser http://localhost:7474, Bolt bolt://localhost:7687, Redis redis://localhost:6379.

## Тестирование

- Директории тестов в репозитории **нет**. `pytest` и `pytest-asyncio` присутствуют в `requirements.txt`, но тесты не написаны — существующего тестового конвейера и CI нет.
- Проверка изменений — ручная: `./update.sh` (пересборка и перезапуск), наблюдение за логами (`docker compose logs -f app`) и за `docker/logs/mentor.log`.
- Если добавляете тесты, используйте `pytest` + `pytest-asyncio` (уже в зависимостях) и кладите их в новую директорию `tests/`.

## Соглашения по коду

- Русский язык для комментариев, лог-сообщений и пользовательских строк — следуйте существующему стилю (эмодзи в логах допустимы, они уже используются).
- Логирование только через `setup_logging()` из `config.py` (логгер `"mentor"`, файл + консоль); `print` используется лишь для ранних сообщений о сети/моделях до инициализации логгера.
- Стиль «эм-дэш» разделителей в комментариях: `# ── Секция ──`.
- Обработка ошибок — широкие `try/except` с логированием и пользовательским фолбэк-сообщением; бот не должен падать на ошибке одного запроса. При правках сохраняйте эту отказоустойчивость (таймауты, fallback через `LLMFactory`, страховочные ответы пользователю).
- В коде есть комментарии-маркеры `ИСПРАВЛЕНО:` от прошлых правок — не удаляйте их без причины, они документируют нетривиальные решения.
- `.gitignore` намеренно исключает черновики (`src/knowledge_base.py`, `src/llm.py`, `src/main.py`, ноутбуки), кэши моделей (`src/hf_models_cache/`, `utils/hf_models_cache/`) и `data/raw/` — не коммитьте их.
- `README.md` частично устарел: упоминает `gref_books/` и `utils/summarizer.py`, которых в репозитории нет. При расхождении доверяйте коду и этому файлу.

## Безопасность

- **Никогда не коммитьте** `.env`, `src/api_keys.env` и `utils/graph_params.json` (в `.gitignore` и `.dockerignore`; секреты не попадают в Docker-образ, передаются через `env_file` в compose).
- Пароль Neo4j задаётся один раз в `.env` (`NEO4J_PASSWORD`): compose собирает из него `NEO4J_AUTH`, а приложение использует тот же пароль — расхождения больше нет. Если меняете пароль на уже проиндексированных данных, старый граф в `docker/neo4j/data` останется под прежним паролем.
- GigaChat создаётся с `verify_ssl_certs=False` (`config/models.json`, модель `gigachat`) — осознанное решение для корпоративной сети; не «чините» без запроса.
- Контейнер приложения работает от `appuser`, `data/` смонтирована read-only — базу знаний меняют только добавлением файлов в `data/` на хосте и рестартом контейнера `app`.
