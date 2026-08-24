#!/bin/bash
# ===========================================
# Запуск Digital Mentor (app + Neo4j + Redis)
# Если приложение уже запущено — только выводит информацию об этом.
# ===========================================

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── Проверка: приложение уже запущено? ──
if docker compose ps --status running --services 2>/dev/null | grep -q "^app$"; then
    log_info "Приложение уже запущено. Текущее состояние контейнеров:"
    docker compose ps
    exit 0
fi

# ── Проверка наличия .env ──
if [ ! -f .env ]; then
    log_error "Файл .env не найден в корне проекта."
    log_error "Создайте его из шаблона: cp .env.example .env — и заполните значения."
    exit 1
fi

# ── Директории для монтирования данных контейнеров ──
mkdir -p docker/neo4j/data docker/neo4j/plugins docker/neo4j/logs docker/redis/data docker/logs

# ── Запуск стека ──
log_info "Запуск Digital Mentor..."
docker compose up -d

log_info "Готово. Состояние контейнеров:"
docker compose ps
log_info "Логи приложения: docker compose logs -f app"
