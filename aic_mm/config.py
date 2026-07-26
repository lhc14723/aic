"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is incomplete or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and attach normalized path metadata."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {config_path}")

    root_value = config.get("project_root", "..")
    project_root = Path(root_value).expanduser()
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()
    config["_config_path"] = config_path
    config["_project_root"] = project_root
    return config


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to the configured project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required mapping from a configuration."""
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration key '{key}' must be a mapping")
    return value
