"""
Фабрика LLM-провайдеров с health-check и упорядоченным fallback.

Загружает конфигурацию моделей из config/models.json, проверяет
доступность default-модели (Cloud.ru) и, при недоступности,
перебирает fallback-модели в заданном порядке.
"""
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.minimax import MiniMax
from langchain_gigachat import GigaChat

from config import setup_logging, MODELS_CONFIG_PATH, MAIN_LLM_TEMPERATURE

logger = setup_logging()


@dataclass
class ResolvedModel:
    """Результат выбора фабрики: готовый LLM + метаданные модели."""

    llm: Any
    model_id: str
    provider: str


class LLMFactory:
    """Создаёт LLM-провайдеров с автоматическим fallback по конфигу."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or MODELS_CONFIG_PATH
        self._config = self._load_config()
        self._models_cache: dict[str, Any] = {}

    def _load_config(self) -> dict:
        """Загружает JSON-конфиг с описанием моделей."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Конфигурация моделей не найдена: {self.config_path}"
            )
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_api_key(self, env_var_name: str) -> str:
        """Получает API-ключ из переменной окружения."""
        key = os.getenv(env_var_name)
        if not key:
            raise ValueError(
                f"Не задан API-ключ: переменная окружения {env_var_name} пуста"
            )
        return key

    def _model_config(self, model_id: str) -> dict:
        """Возвращает конфигурацию конкретной модели по её id."""
        model_cfg = self._config["models"].get(model_id)
        if not model_cfg:
            raise ValueError(f"Модель {model_id} не описана в конфиге")
        return model_cfg

    def _build_llm(self, model_id: str) -> Any:
        """Создаёт объект LLM по описанию из конфига."""
        if model_id in self._models_cache:
            return self._models_cache[model_id]

        model_cfg = self._model_config(model_id)
        provider = model_cfg.get("provider")
        base_url = model_cfg.get("base_url")
        api_key = self._resolve_api_key(model_cfg["api_key_env"])
        model_name = model_cfg.get("model_id")
        temperature = model_cfg.get("temperature", MAIN_LLM_TEMPERATURE)
        context_window = model_cfg.get("context_window")

        if provider == "openai_like":
            llm_kwargs = {
                "api_base": base_url,
                "api_key": api_key,
                "model": model_name,
                "is_chat_model": True,
                "is_function_calling_model": model_cfg.get(
                    "is_function_calling_model", False
                ),
                "temperature": temperature,
                "timeout": model_cfg.get("timeout", 60),
            }
            if context_window:
                llm_kwargs["context_window"] = int(context_window)
            llm = OpenAILike(**llm_kwargs)
        elif provider == "gigachat":
            llm = GigaChat(
                credentials=api_key,
                model=model_name,
                verify_ssl_certs=model_cfg.get("verify_ssl_certs", False),
                scope=model_cfg.get("scope", "GIGACHAT_API_PERS"),
                temperature=temperature,
            )
        elif provider == "minimax":
            llm_kwargs = {
                "model": model_name,
                "api_key": api_key,
                "temperature": temperature,
            }
            if context_window:
                llm_kwargs["context_window"] = int(context_window)
            llm = MiniMax(**llm_kwargs)
        else:
            raise ValueError(f"Неизвестный провайдер: {provider}")

        self._models_cache[model_id] = llm
        return llm

    async def _check_availability(self, model_id: str) -> bool:
        """Проверяет доступность API модели через GET /models."""
        model_cfg = self._model_config(model_id)
        base_url = model_cfg.get("base_url")

        if not base_url:
            # Без base_url проверяем только наличие ключа.
            try:
                self._resolve_api_key(model_cfg["api_key_env"])
                return True
            except ValueError:
                return False

        api_key = self._resolve_api_key(model_cfg["api_key_env"])
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"{base_url}/models", headers=headers
                ) as response:
                    available = response.status == 200
                    if not available:
                        logger.debug(
                            f"[LLM-Factory]: Модель {model_id} вернула HTTP {response.status}"
                        )
                    return available
        except Exception as e:
            logger.debug(
                f"[LLM-Factory]: Модель {model_id} недоступна ({e})"
            )
            return False

    async def get_available_llm(self, exclude: list[str] | None = None) -> ResolvedModel:
        """
        Возвращает первую доступную LLM-модель с метаданными.

        Порядок перебора: default → fallback_order.
        Модели из exclude пропускаются.
        """
        exclude_set = set(exclude or [])
        candidates = [self._config["default"]] + list(
            self._config.get("fallback_order", [])
        )

        for model_id in candidates:
            if model_id in exclude_set:
                continue
            try:
                if await self._check_availability(model_id):
                    model_cfg = self._model_config(model_id)
                    provider = model_cfg.get("provider")
                    logger.info(
                        f"[LLM-Factory]: Выбрана модель {model_id} (провайдер {provider})"
                    )
                    return ResolvedModel(
                        llm=self._build_llm(model_id),
                        model_id=model_id,
                        provider=provider,
                    )
                logger.warning(
                    f"[LLM-Factory]: Модель {model_id} недоступна, пробуем следующую..."
                )
            except Exception as e:
                logger.warning(
                    f"[LLM-Factory]: Ошибка проверки модели {model_id}: {e}"
                )

        raise RuntimeError(
            "Ни одна LLM-модель не доступна. Проверьте конфигурацию и API-ключи."
        )

    def get_default_llm(self) -> Any:
        """Возвращает default LLM без проверки доступности."""
        return self._build_llm(self._config["default"])
