# ===========================================
# Dockerfile для Digital Mentor RAG-бота
# Многоступенчатая сборка: зависимости собираются в builder,
# в runtime копируются только готовые пакеты без build-инструментов.
# ===========================================

# ── Builder: установка Python-зависимостей ──
FROM python:3.11-slim AS builder

WORKDIR /app

# Системные зависимости только для сборки пакетов с нативными расширениями
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости одной командой,
# заменяя torch на CPU-версию (без CUDA, экономит ~1.5–2 ГБ).
# Установка всего списка сразу позволяет pip корректно разрешить
# транзитивные зависимости и избежать бэктрекинга.
RUN pip install --no-cache-dir --upgrade pip && \
    { \
        echo "--extra-index-url https://download.pytorch.org/whl/cpu"; \
        sed 's/^torch==.*/torch==2.12.0+cpu/' requirements.txt; \
    } > /tmp/requirements-build.txt && \
    pip install --no-cache-dir -r /tmp/requirements-build.txt

# Скачиваем модель bge-m3 в кэш заранее, чтобы образ был самодостаточным.
# Используем snapshot_download с конкретной ревизией и исключаем *.safetensors,
# чтобы не тянуть дублирующийся формат и уменьшить размер образа.
RUN mkdir -p /app/src/hf_models_cache && \
    python - <<'PYEOF'
from huggingface_hub import snapshot_download
print("[Build] Скачивание BAAI/bge-m3 (только pytorch_model.bin)...")
snapshot_download(
    repo_id="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    cache_dir="/app/src/hf_models_cache",
    ignore_patterns=["*.safetensors", "*.msgpack", "*.h5", "onnx/**"],
)
print("[Build] Модель BAAI/bge-m3 сохранена в кэш.")
PYEOF

# ── Runtime: финальный минимальный образ ──
FROM python:3.11-slim

# Runtime-зависимости: netcat для entrypoint, libgomp для torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем установленные Python-пакеты и скрипты из builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем исходный код проекта и конфиг LLM-моделей
COPY src/ ./src/
COPY utils/ ./utils/
COPY config/ ./config/

# Копируем заранее скачанную модель из builder
COPY --from=builder /app/src/hf_models_cache /app/src/hf_models_cache

# Копируем entrypoint скрипт
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Создаем необходимые директории
RUN mkdir -p /app/logs

# Создаем непривилегированного пользователя для безопасности.
# chown делаем только для /app/logs — основной код и модель остаются
# под root (доступны на чтение), чтобы не дублировать ~4.5 ГБ в слоях Docker.
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app/logs
USER appuser

# Переменные окружения
# PYTHONPATH=/app/src нужен для плоских импортов внутри src/ (graphrag, config и т.д.)
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Приложение не поднимает HTTP-сервер (long polling бот),
# поэтому стандартный healthcheck на порт не имеет смысла.
HEALTHCHECK NONE

# Точка входа через entrypoint скрипт
ENTRYPOINT ["/app/entrypoint.sh"]
