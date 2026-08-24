"""Проверка, что у каждой модели в конфиге задано положительное окно контекста."""
import json
from pathlib import Path

MODELS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.json"


def test_each_model_has_positive_context_window() -> None:
    config = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
    models = config["models"]
    assert models, "В конфиге нет моделей"

    for model_id, model_cfg in models.items():
        window = model_cfg.get("context_window")
        assert isinstance(window, int), f"{model_id}: context_window должен быть int"
        assert window > 0, f"{model_id}: context_window должен быть положительным"
