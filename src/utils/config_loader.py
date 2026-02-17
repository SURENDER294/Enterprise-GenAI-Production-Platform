"""
config_loader.py

Loads and merges YAML configuration files with environment variable overrides.
Supports multiple environments (dev, staging, prod) via file naming convention.

Usage:
    from src.utils.config_loader import load_config
    cfg = load_config()  # loads configs/config.yaml + env overrides
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Load .env file if it exists (won't override existing env vars)
load_dotenv()


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two dicts. Values in `override` take priority.
    Nested dicts are merged; all other types are replaced.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_dir: Optional[str] = None,
    env: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load YAML config for the given environment.

    Resolution order (later overrides earlier):
      1. configs/config.yaml          <- base config
      2. configs/config.<env>.yaml    <- environment-specific overrides
      3. Environment variables        <- runtime overrides

    Args:
        config_dir: Path to the configs directory. Defaults to ./configs.
        env: Environment name (e.g. "dev", "prod"). Defaults to APP_ENV env var.

    Returns:
        Merged configuration dictionary.
    """
    config_dir = Path(config_dir or "configs")
    env = env or os.getenv("APP_ENV", "development")

    base_config_path = config_dir / "config.yaml"
    env_config_path = config_dir / f"config.{env}.yaml"

    if not base_config_path.exists():
        raise FileNotFoundError(
            f"Base config not found at {base_config_path}. "
            "Make sure you're running from the project root."
        )

    logger.info(f"Loading base config from {base_config_path}")
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Merge environment-specific config if it exists
    if env_config_path.exists():
        logger.info(f"Merging env config from {env_config_path}")
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, env_config)
    else:
        logger.debug(f"No env-specific config found at {env_config_path}, skipping")

    # Allow runtime overrides via env vars using double-underscore as level separator
    # e.g., LLM__TEMPERATURE=0.5 overrides config["llm"]["temperature"]
    _apply_env_overrides(config)

    logger.debug(f"Final config loaded for env='{env}'")
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    """
    Walk all environment variables and apply any that use the
    SECTION__KEY format as overrides to the config dict in-place.
    """
    for env_key, env_val in os.environ.items():
        parts = env_key.lower().split("__")
        if len(parts) < 2:
            # Single-word env vars are not config overrides
            continue
        # Navigate and set nested key
        node = config
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        # Try to cast to original type if possible
        leaf_key = parts[-1]
        if leaf_key in node:
            original_type = type(node[leaf_key])
            try:
                if original_type == bool:
                    node[leaf_key] = env_val.lower() in ("true", "1", "yes")
                elif original_type in (int, float):
                    node[leaf_key] = original_type(env_val)
                else:
                    node[leaf_key] = env_val
            except (ValueError, TypeError):
                node[leaf_key] = env_val
        else:
            node[leaf_key] = env_val
