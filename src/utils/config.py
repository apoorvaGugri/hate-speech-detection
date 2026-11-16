"""Configuration loading and path helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.constants import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigError(Exception):
    """Raised when the configuration file is missing or malformed."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config into a plain nested dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {cfg_path}: {exc}") from exc
    if not isinstance(cfg, dict) or "paths" not in cfg:
        raise ConfigError(f"{cfg_path} must be a mapping with a 'paths' section")
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Turn a config path into an absolute one (relative paths hang off the project root)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_dir(cfg: dict[str, Any], key: str) -> Path:
    """Resolve ``cfg['paths'][key]`` and make sure the directory exists."""
    directory = resolve_path(cfg["paths"][key])
    directory.mkdir(parents=True, exist_ok=True)
    return directory
