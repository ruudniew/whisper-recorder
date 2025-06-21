import json
import threading
from pathlib import Path
from typing import Any, Dict


class ConfigManager:
    """Manages application configuration with persistence and validation"""

    DEFAULT_CONFIG = {
        "hotkey": "control+option+space",
        "audio_device": "default",
        "audio_device_id": None,
        "insertion_method": "keyboard",
        "model": "tiny.en",
        "language": "en",
        "start_at_login": False,
    }

    def __init__(self, config_path: str = "~/.whisper-transcriber/config.json"):
        """Initialize ConfigManager with config file path

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path).expanduser()
        self._lock = threading.Lock()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from disk

        Returns:
            Configuration dictionary
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, use default config
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create config file with defaults
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config = self.DEFAULT_CONFIG.copy()
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=2)
            return default_config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Handle nested keys with dot notation
        if "." in key:
            keys = key.split(".")
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value

        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
        """
        with self._lock:
            # Handle nested keys with dot notation
            if "." in key:
                keys = key.split(".")
                config = self.config
                for k in keys[:-1]:
                    if k not in config or not isinstance(config[k], dict):
                        config[k] = {}
                    config = config[k]
                config[keys[-1]] = value
            else:
                self.config[key] = value

    def save(self) -> None:
        """Persist configuration to disk"""
        with self._lock:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)

    def validate(self) -> bool:
        """Validate current configuration

        Returns:
            True if configuration is valid, False otherwise
        """
        # Check all required keys are present
        for key in self.DEFAULT_CONFIG:
            if key not in self.config:
                return False

        # Type validation
        type_checks = {
            "hotkey": str,
            "audio_device": str,
            "audio_device_id": (int, type(None)),
            "insertion_method": str,
            "model": str,
            "language": str,
            "start_at_login": bool,
        }

        for key, expected_type in type_checks.items():
            if key in self.config and not isinstance(self.config[key], expected_type):
                return False

        return True

    def reset(self) -> None:
        """Reset configuration to defaults"""
        with self._lock:
            self.config = self.DEFAULT_CONFIG.copy()

    def merge(self, new_config: Dict[str, Any]) -> None:
        """Merge new configuration with existing

        Args:
            new_config: Configuration dictionary to merge
        """
        with self._lock:
            self.config.update(new_config)

    def export(self) -> str:
        """Export configuration as JSON string

        Returns:
            JSON string of configuration
        """
        return json.dumps(self.config, indent=2)

    def import_config(self, config_json: str) -> None:
        """Import configuration from JSON string

        Args:
            config_json: JSON string of configuration

        Raises:
            ValueError: If JSON is invalid
        """
        try:
            new_config = json.loads(config_json)
            with self._lock:
                self.config = new_config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON configuration: {e}")
