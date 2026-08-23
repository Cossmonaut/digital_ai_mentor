#!/bin/bash
# ===========================================
# Обновление Digital Mentor
# Пересобирает образ приложения и перезапускает стек.
# ===========================================

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── Проверка наличия .env ──
if [ ! -f .env ]; then
    log_error "Файл .env не найден в корне проекта."
    log_error "Создайте его из шаблона: cp .env.example .env — и заполните значения."
    exit 1
fi

# ── Директории для монтирования (на случай первого запуска) ──
mkdir -p docker/neo4j/data docker/neo4j/plugins docker/neo4j/logs docker/redis/data docker/logs

# ── Обновление базовых образов и пересборка приложения ──
log_info "Обновление базовых образов (neo4j, redis)..."
docker compose pull neo4j redis_cache

log_info "Пересборка образа приложения..."
docker compose build app

# ── Перезапуск стека с новыми образами ──
log_info "Перезапуск контейнеров..."
docker compose up -d

log_info "Обновление завершено. Состояние контейнеров:"
docker compose ps
