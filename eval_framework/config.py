"""
eval_framework.config
=====================
Configuration management for the evaluation suite.
Supports loading YAML configuration and overriding with environment variables.
"""

import os
from typing import Any, Dict, Optional
import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)

_cached_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load configuration from YAML file and apply environment variable overrides.
    """
    global _cached_config
    if _cached_config is not None and not force_reload and config_path is None:
        return _cached_config

    path = config_path or os.environ.get("EVAL_CONFIG_PATH", DEFAULT_CONFIG_PATH)

    config: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Failed to load config from {path}: {e}. Using defaults.")

    # Environment variable overrides
    if "DEFAULT_PROVIDER" in os.environ:
        config.setdefault("shared", {})["PROVIDER"] = os.environ["DEFAULT_PROVIDER"]
    if "OLLAMA_BASE_URL" in os.environ:
        config.setdefault("shared", {})["OLLAMA_URL"] = os.environ["OLLAMA_BASE_URL"]

    _cached_config = config
    return config


def get_section(section_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve a specific section with fallback to shared configuration."""
    cfg = load_config(config_path)
    shared = cfg.get("shared", {})
    section = cfg.get(section_name, {})

    merged = dict(shared)
    merged.update(section)
    return merged
