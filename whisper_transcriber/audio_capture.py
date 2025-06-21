import logging
import threading
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd

from .models import AudioDevice, AudioConfig


logger = logging.getLogger(__name__)


class AudioCapture:
    """Handles audio input and streaming"""

    def __init__(self, device_id: Optional[int] = None):
        """Initialize AudioCapture with optional device ID

        Args:
            device_id: Specific audio device ID to use, None for default
        """
        self.device_id = device_id
        self.stream = None
        self.is_recording = False
        self.audio_callback = None
        self._lock = threading.Lock()

        # Audio configuration matching WhisperLiveKit requirements
        self._audio_config = AudioConfig()

    def list_devices(self) -> List[AudioDevice]:
        """Get available audio input devices

        Returns:
            List of AudioDevice objects
        """
        devices = []
        all_devices = sd.query_devices()
        default_input = sd.default.device[0] if sd.default.device else None

        for idx, device in enumerate(all_devices):
            # Only include devices with input channels
            if device["max_input_channels"] > 0:
                devices.append(
                    AudioDevice(
                        id=idx,
                        name=device["name"],
                        channels=device["max_input_channels"],
                        sample_rate=int(device["default_samplerate"]),
                        is_default=(idx == default_input),
                    )
                )

        return devices

    def start_recording(self, callback: Callable[[bytes], None]) -> bool:
        """Begin audio capture with callback for chunks

        Args:
            callback: Function to call with audio data chunks

        Returns:
            True if recording started successfully, False otherwise
        """
        with self._lock:
            if self.is_recording:
                logger.warning("Already recording")
                return False

            try:
                self.audio_callback = callback
                self.stream = sd.InputStream(
                    device=self.device_id,
                    channels=self._audio_config.channels,
                    samplerate=self._audio_config.sample_rate,
                    blocksize=self._audio_config.chunk_size,
                    dtype=self._audio_config.format,
                    callback=self._audio_callback,
                )
                self.stream.start()
                self.is_recording = True
                logger.info(f"Started recording on device {self.device_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to start recording: {e}")
                self.stream = None
                self.audio_callback = None
                return False

    def stop_recording(self) -> None:
        """Stop audio capture and cleanup"""
        with self._lock:
            if not self.is_recording:
                return

            self.is_recording = False

            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}")
                finally:
                    self.stream = None

            self.audio_callback = None
            logger.info("Stopped recording")

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """Internal callback processing audio chunks

        Args:
            indata: Audio data as numpy array
            frames: Number of frames
            time: Timing information
            status: Status flags
        """
        if status:
            logger.warning(f"Audio callback error: {status}")

        if self.is_recording and self.audio_callback:
            # Convert numpy array to bytes
            audio_bytes = indata.tobytes()
            # Log only occasionally to avoid spam
            if hasattr(self, "_audio_log_counter"):
                self._audio_log_counter += 1
            else:
                self._audio_log_counter = 0

            if self._audio_log_counter % 100 == 0:  # Log every 100th chunk
                logger.debug(
                    f"Audio captured: {len(audio_bytes)} bytes, max amplitude: {np.max(np.abs(indata))}"
                )

            try:
                self.audio_callback(audio_bytes)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")

    def get_current_device(self) -> Optional[AudioDevice]:
        """Get information about the current audio device

        Returns:
            AudioDevice object or None if using default
        """
        devices = self.list_devices()

        if self.device_id is not None:
            # Find specific device
            for device in devices:
                if device.id == self.device_id:
                    return device
        else:
            # Return default device
            for device in devices:
                if device.is_default:
                    return device

        return None

    def set_device(self, device_id: Optional[int]) -> None:
        """Set the audio input device

        Args:
            device_id: Device ID to use, None for default

        Raises:
            RuntimeError: If called while recording
        """
        if self.is_recording:
            raise RuntimeError("Cannot change device while recording")

        self.device_id = device_id

    def get_audio_config(self) -> AudioConfig:
        """Get current audio configuration

        Returns:
            AudioConfig object with current settings
        """
        return self._audio_config
