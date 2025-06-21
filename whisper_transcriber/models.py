from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class AudioConfig:
    """Audio capture configuration settings"""

    sample_rate: int = 16000  # WhisperLiveKit default
    channels: int = 1  # Mono
    chunk_size: int = 512  # Smaller chunks for lower latency (32ms)
    format: str = "int16"  # 16-bit PCM


@dataclass(frozen=True)
class ServerConfig:
    """WhisperLiveKit server configuration"""

    host: str = "localhost"
    port: int = 9090
    model: str = "tiny.en"  # Start with tiny for low latency
    language: str = "en"
    vad_enabled: bool = True
    use_gpu: bool = True

    @property
    def websocket_url(self) -> str:
        """Generate WebSocket URL from host and port"""
        return f"ws://{self.host}:{self.port}/asr"


@dataclass
class AudioDevice:
    """Represents an audio input device"""

    id: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False

    def __str__(self) -> str:
        """String representation of audio device"""
        base = f"{self.name} (ID: {self.id}, {self.channels}ch @ {self.sample_rate}Hz)"
        if self.is_default:
            return f"{base} [DEFAULT]"
        return base


class InsertMethod(Enum):
    """Text insertion method options"""

    CLIPBOARD = "clipboard"  # Copy/paste approach
    KEYBOARD = "keyboard"  # Direct typing
    AUTO = "auto"  # Choose best method
