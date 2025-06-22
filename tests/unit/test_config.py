import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from whisper_transcriber.config import ConfigManager


class TestConfigManager:
    """Test suite for ConfigManager class"""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test_key": "test_value"}, f)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def config_manager(self, temp_config_file):
        """Create a ConfigManager instance with temp config file"""
        return ConfigManager(config_path=temp_config_file)

    def test_init_loads_existing_config(self, temp_config_file):
        """Test that ConfigManager loads existing configuration on init"""
        # Write test config
        test_config = {"key1": "value1", "key2": 42}
        with open(temp_config_file, 'w') as f:
            json.dump(test_config, f)
        
        # Create manager and verify config loaded
        manager = ConfigManager(config_path=temp_config_file)
        assert manager.config == test_config

    def test_init_creates_config_if_not_exists(self):
        """Test that ConfigManager creates default config if file doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            
            manager = ConfigManager(config_path=str(config_path))
            
            # Should create file with default config
            assert config_path.exists()
            assert manager.config == manager.DEFAULT_CONFIG

    def test_init_creates_parent_directories(self):
        """Test that ConfigManager creates parent directories if they don't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nested" / "dirs" / "config.json"
            
            manager = ConfigManager(config_path=str(config_path))
            
            assert config_path.exists()
            assert config_path.parent.exists()

    def test_get_returns_value_for_existing_key(self, config_manager):
        """Test get() returns correct value for existing key"""
        config_manager.config = {"test_key": "test_value"}
        assert config_manager.get("test_key") == "test_value"

    def test_get_returns_default_for_missing_key(self, config_manager):
        """Test get() returns default value for missing key"""
        assert config_manager.get("missing_key", "default") == "default"
        assert config_manager.get("missing_key") is None

    def test_get_handles_nested_keys(self, config_manager):
        """Test get() can retrieve nested configuration values"""
        config_manager.config = {
            "level1": {
                "level2": {
                    "value": "nested"
                }
            }
        }
        assert config_manager.get("level1.level2.value") == "nested"
        assert config_manager.get("level1.level2") == {"value": "nested"}

    def test_set_updates_existing_key(self, config_manager):
        """Test set() updates existing configuration key"""
        config_manager.config = {"key": "old_value"}
        config_manager.set("key", "new_value")
        assert config_manager.config["key"] == "new_value"

    def test_set_creates_new_key(self, config_manager):
        """Test set() creates new configuration key"""
        config_manager.set("new_key", "new_value")
        assert config_manager.config["new_key"] == "new_value"

    def test_set_handles_nested_keys(self, config_manager):
        """Test set() can update nested configuration values"""
        config_manager.config = {"level1": {}}
        config_manager.set("level1.level2.value", "nested")
        assert config_manager.config["level1"]["level2"]["value"] == "nested"

    def test_save_writes_config_to_file(self, config_manager, temp_config_file):
        """Test save() persists configuration to disk"""
        config_manager.config = {"saved_key": "saved_value"}
        config_manager.save()
        
        # Read file and verify
        with open(temp_config_file, 'r') as f:
            saved_config = json.load(f)
        assert saved_config == {"saved_key": "saved_value"}

    def test_save_creates_directories_if_needed(self):
        """Test save() creates parent directories if they don't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "new_dir" / "config.json"
            manager = ConfigManager(config_path=str(config_path))
            
            # Delete the config file and directory created by init
            config_path.unlink()
            config_path.parent.rmdir()
            
            manager.save()
            assert config_path.exists()

    def test_default_config_contains_required_keys(self):
        """Test that default config contains all required keys"""
        manager = ConfigManager(config_path="dummy_path")
        required_keys = [
            "hotkey", "audio_device", "audio_device_id", "insertion_method",
            "model", "language", "start_at_login", "vad_enabled", "use_gpu"
        ]
        for key in required_keys:
            assert key in manager.DEFAULT_CONFIG

    def test_default_config_values(self):
        """Test that default config has correct values"""
        manager = ConfigManager(config_path="dummy_path")
        assert manager.DEFAULT_CONFIG["hotkey"] == "control+option+space"
        assert manager.DEFAULT_CONFIG["audio_device"] == "default"
        assert manager.DEFAULT_CONFIG["audio_device_id"] is None
        assert manager.DEFAULT_CONFIG["insertion_method"] == "keyboard"
        assert manager.DEFAULT_CONFIG["model"] == "tiny.en"
        assert manager.DEFAULT_CONFIG["language"] == "en"
        assert manager.DEFAULT_CONFIG["start_at_login"] is False
        assert manager.DEFAULT_CONFIG["vad_enabled"] is False
        assert manager.DEFAULT_CONFIG["use_gpu"] is False

    def test_handles_corrupted_config_file(self):
        """Test graceful handling of corrupted config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json{")
            temp_path = f.name
        
        try:
            manager = ConfigManager(config_path=temp_path)
            # Should fall back to default config
            assert manager.config == manager.DEFAULT_CONFIG
        finally:
            os.unlink(temp_path)

    def test_validate_config_valid(self, config_manager):
        """Test validate() passes for valid configuration"""
        config_manager.config = config_manager.DEFAULT_CONFIG
        assert config_manager.validate() is True

    def test_validate_config_invalid_type(self, config_manager):
        """Test validate() fails for invalid type"""
        config_manager.config = {"hotkey": 123}  # Should be string
        assert config_manager.validate() is False

    def test_validate_config_missing_required(self, config_manager):
        """Test validate() fails for missing required keys"""
        config_manager.config = {"hotkey": "cmd+shift+r"}  # Missing other keys
        assert config_manager.validate() is False

    def test_reset_to_defaults(self, config_manager):
        """Test reset() restores default configuration"""
        config_manager.config = {"custom": "value"}
        config_manager.reset()
        assert config_manager.config == config_manager.DEFAULT_CONFIG

    def test_merge_configs(self, config_manager):
        """Test merge() combines configurations correctly"""
        config_manager.config = {"key1": "value1", "key2": "old"}
        new_config = {"key2": "new", "key3": "value3"}
        config_manager.merge(new_config)
        
        assert config_manager.config == {
            "key1": "value1",
            "key2": "new",
            "key3": "value3"
        }

    def test_export_config(self, config_manager):
        """Test export() returns config as JSON string"""
        config_manager.config = {"key": "value"}
        exported = config_manager.export()
        assert json.loads(exported) == {"key": "value"}

    def test_import_config(self, config_manager):
        """Test import_config() loads configuration from JSON string"""
        config_json = '{"imported": "config"}'
        config_manager.import_config(config_json)
        assert config_manager.config == {"imported": "config"}

    def test_import_invalid_json(self, config_manager):
        """Test import_config() handles invalid JSON gracefully"""
        with pytest.raises(ValueError):
            config_manager.import_config("invalid json{")

    def test_config_path_expansion(self):
        """Test that config path with ~ is expanded correctly"""
        manager = ConfigManager(config_path="~/test_config.json")
        assert str(manager.config_path).startswith(str(Path.home()))

    def test_thread_safety_multiple_writes(self, config_manager):
        """Test that concurrent writes are handled safely"""
        import threading
        
        def writer(key, value):
            for _ in range(100):
                config_manager.set(key, value)
                config_manager.save()
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=writer, args=(f"key{i}", f"value{i}"))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All keys should be present
        for i in range(5):
            assert f"key{i}" in config_manager.config