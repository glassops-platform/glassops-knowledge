# settings.py
"""
Loads config/config.json and validates the model settings.

Every model id the package uses comes from this file. No module hard-codes one.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.json"
PROMPTS_PATH = CONFIG_DIR / "prompts.yml"


class ConfigError(Exception):
    """The config file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str
    dimensions: int
    batch_size: int
    rpm_limit: int
    tpm_limit: int


@dataclass(frozen=True)
class GenerationSettings:
    docs_model: str
    query_model: str
    max_output_tokens: int
    temperature: float
    rpm_limit: int
    tpm_limit: int


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the config file. Raises ConfigError instead of falling back to defaults."""
    config_path = Path(path) if path else CONFIG_PATH
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {config_path}") from e
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"could not read config file {config_path}: {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError(f"config file {config_path} must hold a JSON object")
    return cfg


def _models(cfg: Dict[str, Any]) -> Dict[str, Any]:
    models = cfg.get("models")
    if not isinstance(models, dict):
        raise ConfigError('config is missing the "models" object')
    return models


def _require_str(section: Dict[str, Any], key: str, where: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f'config "{where}.{key}" must be a non-empty model id')
    return value.strip()


def _require_positive_int(section: Dict[str, Any], key: str, where: str, default: Optional[int] = None) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f'config "{where}.{key}" must be a positive integer')
    return value


def embedding_settings(cfg: Dict[str, Any]) -> EmbeddingSettings:
    """
    The embedding model and its output size.

    The dimensions value is required for every model. Some embedding models default to 3072
    dimensions when none is sent, so leaving it implicit would change the index shape.
    """
    section = _models(cfg).get("embedding")
    if not isinstance(section, dict):
        raise ConfigError('config is missing "models.embedding"')
    where = "models.embedding"
    return EmbeddingSettings(
        model=_require_str(section, "id", where),
        dimensions=_require_positive_int(section, "dimensions", where),
        batch_size=_require_positive_int(section, "batch_size", where, default=10),
        rpm_limit=_require_positive_int(section, "rpm_limit", where, default=90),
        tpm_limit=_require_positive_int(section, "tpm_limit", where, default=25000),
    )


def generation_settings(cfg: Dict[str, Any]) -> GenerationSettings:
    """The generation models: one for documentation, one for RAG answers."""
    section = _models(cfg).get("generation")
    if not isinstance(section, dict):
        raise ConfigError('config is missing "models.generation"')
    where = "models.generation"
    temperature = section.get("temperature", 0.2)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ConfigError(f'config "{where}.temperature" must be a number')
    return GenerationSettings(
        docs_model=_require_str(section, "docs", where),
        query_model=_require_str(section, "query", where),
        max_output_tokens=_require_positive_int(section, "max_output_tokens", where, default=8192),
        temperature=float(temperature),
        rpm_limit=_require_positive_int(section, "rpm_limit", where, default=24),
        tpm_limit=_require_positive_int(section, "tpm_limit", where, default=12000),
    )


def persist_dir(cfg: Dict[str, Any]) -> Path:
    """The Chroma index folder, resolved against the package root."""
    name = cfg.get("vector_store", {}).get("persist_dir", "glassops_index")
    path = Path(name)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def manifest_path(cfg: Dict[str, Any]) -> Path:
    """The committed index manifest, resolved against the package root."""
    name = cfg.get("index", {}).get("manifest", "index/manifest.json")
    path = Path(name)
    return path if path.is_absolute() else PACKAGE_ROOT / path


def collection_name(cfg: Dict[str, Any]) -> str:
    return cfg.get("vector_store", {}).get("collection", "glassops_knowledge")
