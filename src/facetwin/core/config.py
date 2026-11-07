"""Configuration loading and validation for FaceTwin."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from facetwin.core.exceptions import ConfigurationError

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads a YAML or JSON configuration file.

    Args:
        config_path: Path to the configuration file (.yaml, .yml, or .json).

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        ConfigurationError: If file is missing, format is unsupported, or parsing fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path.resolve()}")

    if path.suffix.lower() == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Error parsing JSON file {path}: {e}") from e

    if not HAS_YAML:
        raise ConfigurationError(
            "PyYAML is not installed. Please install 'pyyaml' to parse YAML configuration files."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except Exception as e:
        raise ConfigurationError(f"Error parsing YAML file {path}: {e}") from e


class ConfigManager:
    """Manages project configurations across default, data, and model configs."""

    def __init__(self, base_config_dir: str | Path = "configs"):
        self.base_dir = Path(base_config_dir)
        self._default_config: Optional[Dict[str, Any]] = None

    def get_default_config(self) -> Dict[str, Any]:
        """Loads and caches the default configuration."""
        if self._default_config is None:
            default_path = self.base_dir / "default.yaml"
            self._default_config = load_yaml_config(default_path)
        return self._default_config

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Loads preprocessing settings."""
        return load_yaml_config(self.base_dir / "data_configs" / "preprocessing.yaml")

    def get_features_config(self) -> Dict[str, Any]:
        """Loads feature extraction parameters."""
        return load_yaml_config(self.base_dir / "data_configs" / "features.yaml")

    def get_ml_config(self) -> Dict[str, Any]:
        """Loads classical ML pipeline settings."""
        return load_yaml_config(self.base_dir / "model_configs" / "ml_pipeline.yaml")

    def get_lstm_config(self) -> Dict[str, Any]:
        """Loads LSTM sequence model settings."""
        return load_yaml_config(self.base_dir / "model_configs" / "lstm_model.yaml")

    def get_kalman_config(self) -> Dict[str, Any]:
        """Loads Kalman filter state estimator settings."""
        return load_yaml_config(self.base_dir / "model_configs" / "kalman_filter.yaml")
