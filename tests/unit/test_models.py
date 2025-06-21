import pytest
from dataclasses import asdict

from whisper_transcriber.models import AudioConfig, ServerConfig, AudioDevice, InsertMethod


class TestAudioConfig:
    """Test suite for AudioConfig dataclass"""
    
    def test_default_values(self):
        """Test AudioConfig has correct default values"""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 512
        assert config.format == "int16"
    
    def test_custom_values(self):
        """Test AudioConfig can be initialized with custom values"""
        config = AudioConfig(
            sample_rate=48000,
            channels=2,
            chunk_size=2048,
            format="float32"
        )
        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.chunk_size == 2048
        assert config.format == "float32"
    
    def test_to_dict(self):
        """Test AudioConfig can be converted to dictionary"""
        config = AudioConfig()
        config_dict = asdict(config)
        
        assert config_dict == {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 512,
            "format": "int16"
        }
    
    def test_immutability(self):
        """Test AudioConfig is frozen (immutable)"""
        config = AudioConfig()
        with pytest.raises(AttributeError):
            config.sample_rate = 48000


class TestServerConfig:
    """Test suite for ServerConfig dataclass"""
    
    def test_default_values(self):
        """Test ServerConfig has correct default values"""
        config = ServerConfig()
        assert config.host == "localhost"
        assert config.port == 9090
        assert config.model == "tiny.en"
        assert config.language == "en"
        assert config.vad_enabled is True
        assert config.use_gpu is True
    
    def test_custom_values(self):
        """Test ServerConfig can be initialized with custom values"""
        config = ServerConfig(
            host="127.0.0.1",
            port=8080,
            model="base",
            language="es",
            vad_enabled=False,
            use_gpu=False
        )
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.model == "base"
        assert config.language == "es"
        assert config.vad_enabled is False
        assert config.use_gpu is False
    
    def test_to_dict(self):
        """Test ServerConfig can be converted to dictionary"""
        config = ServerConfig()
        config_dict = asdict(config)
        
        assert config_dict == {
            "host": "localhost",
            "port": 9090,
            "model": "tiny.en",
            "language": "en",
            "vad_enabled": True,
            "use_gpu": True,
        }
    
    def test_immutability(self):
        """Test ServerConfig is frozen (immutable)"""
        config = ServerConfig()
        with pytest.raises(AttributeError):
            config.host = "192.168.1.1"
    
    def test_websocket_url(self):
        """Test ServerConfig can generate WebSocket URL"""
        config = ServerConfig(host="localhost", port=9090)
        assert config.websocket_url == "ws://localhost:9090/asr"
        
        config = ServerConfig(host="127.0.0.1", port=8080)
        assert config.websocket_url == "ws://127.0.0.1:8080/asr"


class TestAudioDevice:
    """Test suite for AudioDevice dataclass"""
    
    def test_creation(self):
        """Test AudioDevice can be created with required fields"""
        device = AudioDevice(
            id=0,
            name="Built-in Microphone",
            channels=2,
            sample_rate=44100
        )
        assert device.id == 0
        assert device.name == "Built-in Microphone"
        assert device.channels == 2
        assert device.sample_rate == 44100
    
    def test_optional_fields(self):
        """Test AudioDevice optional fields"""
        device = AudioDevice(
            id=1,
            name="USB Microphone",
            channels=1,
            sample_rate=48000,
            is_default=True
        )
        assert device.is_default is True
        
        device2 = AudioDevice(
            id=2,
            name="Another Mic",
            channels=2,
            sample_rate=44100
        )
        assert device2.is_default is False
    
    def test_string_representation(self):
        """Test AudioDevice string representation"""
        device = AudioDevice(
            id=0,
            name="Built-in Microphone",
            channels=2,
            sample_rate=44100
        )
        assert str(device) == "Built-in Microphone (ID: 0, 2ch @ 44100Hz)"
        
        device_default = AudioDevice(
            id=1,
            name="USB Mic",
            channels=1,
            sample_rate=48000,
            is_default=True
        )
        assert str(device_default) == "USB Mic (ID: 1, 1ch @ 48000Hz) [DEFAULT]"
    
    def test_equality(self):
        """Test AudioDevice equality comparison"""
        device1 = AudioDevice(id=0, name="Mic", channels=2, sample_rate=44100)
        device2 = AudioDevice(id=0, name="Mic", channels=2, sample_rate=44100)
        device3 = AudioDevice(id=1, name="Mic", channels=2, sample_rate=44100)
        
        assert device1 == device2
        assert device1 != device3


class TestInsertMethod:
    """Test suite for InsertMethod enum"""
    
    def test_enum_values(self):
        """Test InsertMethod has correct values"""
        assert InsertMethod.CLIPBOARD.value == "clipboard"
        assert InsertMethod.KEYBOARD.value == "keyboard"
        assert InsertMethod.AUTO.value == "auto"
    
    def test_enum_members(self):
        """Test all InsertMethod members are accessible"""
        methods = list(InsertMethod)
        assert len(methods) == 3
        assert InsertMethod.CLIPBOARD in methods
        assert InsertMethod.KEYBOARD in methods
        assert InsertMethod.AUTO in methods
    
    def test_from_string(self):
        """Test InsertMethod can be created from string"""
        assert InsertMethod("clipboard") == InsertMethod.CLIPBOARD
        assert InsertMethod("keyboard") == InsertMethod.KEYBOARD
        assert InsertMethod("auto") == InsertMethod.AUTO
    
    def test_invalid_value(self):
        """Test InsertMethod raises error for invalid value"""
        with pytest.raises(ValueError):
            InsertMethod("invalid_method")