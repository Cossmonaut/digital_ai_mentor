# Digital Mentor - ИИ-ментор на базе RAG и GraphRAG

Telegram-бот для корпоративного менторинга, использующий GraphRAG (Neo4j + LlamaIndex) с каскадной системой LLM (Claude → GigaChat → MiniMax).

## 🏗 Архитектура

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Telegram Bot   │────▶│  GraphRAG Engine │────▶│   Neo4j Graph   │
│   (aiogram 3)   │     │  (LlamaIndex)    │     │   Database      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │       ▲
                              ▼       │
                       ┌──────────────────┐     ┌─────────────────┐
                       │   Redis Cache    │     │  LLM Providers  │
                       │  (History +      │     │  • Cloud.ru     │
                       │   Answers)       │     │  • GigaChat     │
                       └──────────────────┘     │  • MiniMax      │
                                                └─────────────────┘
```

## 📋 Требования

- Docker 24.0+
- Docker Compose 2.20+
- 4+ GB RAM (рекомендуется 8+ GB)
- 10+ GB свободного места на диске
- Доступ к API: Cloud.ru, GigaChat, MiniMax, Telegram Bot API

## 🚀 Быстрый старт

### 1. Клонирование и настройка

```bash
git clone <repository-url>
cd digital_mentor
```

### 2. Настройка переменных окружения

```bash
# Скопируйте шаблон и заполните своими ключами
cp .env.example src/api_keys.env

# Отредактируйте файл с вашими реальными ключами
nano src/api_keys.env
```

**Обязательные переменные в `src/api_keys.env`:**
```env
BOT_TOKEN=your_telegram_bot_token
CLOUD_RU_API_KEY=your_cloud_ru_key
GIGACHAT_API_KEY=your_gigachat_key
MINIMAX_API_KEY=your_minimax_key
NEO4J_PASSWORD=secure_password_here
```

### 3. Создание внешних томов Docker (один раз)

```bash
# Создаем тома для персистентности Neo4j
docker volume create neo4j_data
docker volume create neo4j_plugins
```

### 4. Запуск

```bash
# Из корня проекта
docker-compose -f utils/docker-compose.yaml up -d --build
```

### 5. Проверка статуса

```bash
# Логи приложения
docker-compose -f utils/docker-compose.yaml logs -f app

# Статус контейнеров
docker-compose -f utils/docker-compose.yaml ps
```

## 🔧 Конфигурация

### Основные файлы конфигурации

| Файл | Описание |
|------|----------|
| `src/config.py` | Централизованная конфигурация Python |
| `src/api_keys.env` | Секреты и API ключи (НЕ коммитить!) |
| `utils/docker-compose.yaml` | Оркестрация контейнеров |
| `utils/prompts/prompts_rag.yaml` | Промпты для основной LLM (Claude) |
| `utils/prompts/prompts_rag_gigachat.yaml` | Промпты для резервной LLM (GigaChat) |

### Переменные окружения (config.py)

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `NEO4J_URL` | `bolt://neo4j:7687` | Адрес Neo4j в Docker сети |
| `REDIS_URL` | `redis://redis_cache:6379` | Адрес Redis в Docker сети |
| `MAIN_LLM_MODEL` | `anthropic/claude-sonnet-4.6` | Модель основного LLM |
| `MAIN_LLM_BASE` | `https://foundation-models.api.cloud.ru/v1` | Endpoint Cloud.ru |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `CACHE_TTL_SECONDS` | `3600` | TTL кеша ответов (1 час) |
| `REDIS_TTL` | `86400` | TTL истории диалогов (24 часа) |

## 📦 Структура проекта

```
digital_mentor/
├── .env.example              # Шаблон переменных окружения
├── .dockerignore             # Исключения для Docker build
├── .gitignore                # Исключения для Git
├── Dockerfile                # Образ приложения
├── requirements.txt          # Python зависимости
├── README.md                 # Эта документация
├── data/                     # Текстовые файлы базы знаний (read-only в контейнере)
│   ├── summary_*.txt         # Конспекты книг
│   └── ...
├── gref_books/               # Исходные книги (НЕ в Docker, только для обработки)
├── logs/                     # Логи приложения (volume)
├── src/                      # Основной код приложения
│   ├── main_rag.py           # Точка входа (Telegram бот)
│   ├── graphrag.py           # GraphRAG движок
│   ├── graph_knowledge_base.py # Работа с Neo4j
│   ├── config.py             # Конфигурация
│   └── api_keys.env          # Секреты (создается из .env.example)
└── utils/                    # Утилиты и конфигурация Docker
    ├── docker-compose.yaml   # Docker Compose файл
    ├── summarizer.py         # Скрипт суммаризации книг
    ├── parse_pdf.py          # Парсинг PDF
    └── prompts/              # YAML промпты
        ├── prompts_rag.yaml
        └── prompts_rag_gigachat.yaml
```

## 🔄 Обновление базы знаний

База знаний хранится в Neo4j. Текстовые файлы в `data/` используются только для **первичной индексации** или **добавления новых документов**.

### Добавление новых документов

1. Положите `.txt` файлы в папку `data/`
2. Перезапустите контейнер приложения:
   ```bash
   docker-compose -f utils/docker-compose.yaml restart app
   ```
3. При старте `GraphKnowledgeBase.init_index()` автоматически обнаружит новые файлы и добавит их в граф.

### Полная переиндексация

```bash
# Остановите приложение
docker-compose -f utils/docker-compose.yaml stop app

# Очистите базу Neo4j (через веб-интерфейс :7474 или cypher-shell)
# MATCH (n) DETACH DELETE n

# Запустите приложение заново
docker-compose -f utils/docker-compose.yaml start app
```

## 🛠 Полезные команды

```bash
# Просмотр логов в реальном времени
docker-compose -f utils/docker-compose.yaml logs -f app

# Вход в контейнер приложения
docker-compose -f utils/docker-compose.yaml exec app bash

# Вход в Neo4j Cypher Shell
docker-compose -f utils/docker-compose.yaml exec neo4j cypher-shell -u neo4j -p password123

# Пересборка только приложения
docker-compose -f utils/docker-compose.yaml build app

# Полная остановка с удалением контейнеров (данные в volumes сохраняются)
docker-compose -f utils/docker-compose.yaml down

# Полная очистка (ВНИМАНИЕ: удаляет volumes с данными!)
docker-compose -f utils/docker-compose.yaml down -v
```

## 🌐 Доступ к сервисам

| Сервис | Порт | URL |
|--------|------|-----|
| Neo4j Browser | 7474 | http://localhost:7474 |
| Neo4j Bolt | 7687 | bolt://localhost:7687 |
| Redis | 6379 | redis://localhost:6379 |

**Neo4j логин:** `neo4j` / **пароль:** из `NEO4J_PASSWORD` (по умолчанию `password123` в docker-compose)

## 🔒 Безопасность

- **Никогда не коммитьте** `src/api_keys.env` — он в `.gitignore`
- Используйте сильные пароли для Neo4j
- В продакшене настройте файрвол: открывайте только нужные порты
- Рассмотрите использование Docker Secrets для продакшена
- Регулярно обновляйте базовые образы: `docker pull python:3.11-slim neo4j:5.26.0 redis:7-alpine`

## 🐛 Troubleshooting

### Бот не запускается
```bash
# Проверьте логи
docker-compose -f utils/docker-compose.yaml logs app

# Частые причины:
# 1. Не заполнен src/api_keys.env
# 2. Недоступен Telegram API (нужен PROXY_URL)
# 3. Неверные API ключи
```

### Neo4j не запускается
```bash
# Проверьте логи
docker-compose -f utils/docker-compose.yaml logs neo4j

# Убедитесь, что тома созданы
docker volume ls | grep neo4j
```

### Ошибки памяти (OOM)
- Увеличьте лимиты Docker Desktop (Settings → Resources → Advanced)
- Для Neo4j настройте heap в docker-compose:
  ```yaml
  environment:
    - NEO4J_dbms_memory_heap_max__size=2G
  ```

### Модели не скачиваются (offline режим)
Проект настроен на offline-использование `bge-m3`. Убедитесь, что модель заранее скачана в `src/hf_models_cache/` или доступен интернет при первом запуске.

## 📝 Лицензия

Внутренний проект. Все права защищены.

---

**Поддержка:** При возникновении проблем проверьте логи (`docker-compose logs -f app`) и убедитесь, что все API ключи валидны.