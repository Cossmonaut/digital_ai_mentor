#!/bin/bash
# ===========================================
# Остановка Digital Mentor
# Данные в ./docker/ сохраняются.
# ===========================================

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

if ! docker compose ps --status running --services 2>/dev/null | grep -q .; then
    log_warn "Приложение не запущено — останавливать нечего."
    exit 0
fi

log_info "Остановка Digital Mentor..."
docker compose down
log_info "Приложение остановлено. Данные в ./docker/ сохранены."
