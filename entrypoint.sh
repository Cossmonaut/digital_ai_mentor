#!/bin/bash
# ===========================================
# Entrypoint скрипт для Digital Mentor
# Ожидает готовность зависимостей и запускает приложение
# ===========================================

set -e

# Цвета для логов
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Функция ожидания готовности сервиса
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=${4:-30}
    local attempt=1

    log_info "Ожидание готовности $service_name ($host:$port)..."

    while ! nc -z "$host" "$port" >/dev/null 2>&1; do
        if [ $attempt -ge $max_attempts ]; then
            log_error "$service_name недоступен после $max_attempts попыток"
            return 1
        fi
        log_warn "Попытка $attempt/$max_attempts: $service_name еще не готов, ждем 2 сек..."
        sleep 2
        attempt=$((attempt + 1))
    done

    log_info "$service_name готов!"
    return 0
}

# Проверка наличия netcat
if ! command -v nc &> /dev/null; then
    log_warn "netcat не установлен, устанавливаем..."
    apt-get update && apt-get install -y netcat-openbsd >/dev/null 2>&1
fi

# ===========================================
# ПРОВЕРКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ
# ===========================================

log_info "Проверка критических переменных окружения..."

REQUIRED_VARS=(
    "BOT_TOKEN"
    "CLOUD_RU_API_KEY"
    "GIGACHAT_API_KEY"
    "MINIMAX_API_KEY"
    "NEO4J_PASSWORD"
)

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    log_error "Отсутствуют обязательные переменные окружения:"
    for var in "${MISSING_VARS[@]}"; do
        log_error "  - $var"
    done
    log_error "Заполните файл .env в корне проекта на основе .env.example"
    exit 1
fi

log_info "Все критические переменные окружения заданы"

# ===========================================
# ОЖИДАНИЕ ЗАВИСИМОСТЕЙ
# ===========================================

# Neo4j
wait_for_service "neo4j" "7687" "Neo4j Bolt" 60

# Redis
wait_for_service "redis_cache" "6379" "Redis" 30

# Дополнительная пауза для полной инициализации Neo4j
log_info "Дополнительная пауза для стабилизации Neo4j (5 сек)..."
sleep 5

# ===========================================
# ПРОВЕРКА БАЗЫ ЗНАНИЙ
# ===========================================

log_info "Проверка наличия данных в /app/data..."
if [ ! -d "/app/data" ] || [ -z "$(ls -A /app/data/*.txt 2>/dev/null)" ]; then
    log_warn "Папка /app/data пуста или не содержит .txt файлов"
    log_warn "База знаний будет пустой до добавления документов"
else
    FILE_COUNT=$(ls -1 /app/data/*.txt 2>/dev/null | wc -l)
    log_info "Найдено $FILE_COUNT текстовых файлов в базе знаний"
fi

# ===========================================
# СОЗДАНИЕ ДИРЕКТОРИЙ
# ===========================================

mkdir -p /app/logs /app/src/hf_models_cache

# ===========================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ===========================================

log_info "Запуск Digital Mentor..."
log_info "Python версия: $(python --version)"
log_info "Рабочая директория: $(pwd)"
log_info "PYTHONPATH: $PYTHONPATH"

# Передаем управление основному процессу
exec python -m src.main_rag