import logging
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import threading
import time

from whisper_transcriber.audio_capture import AudioCapture
from whisper_transcriber.models import AudioDevice, AudioConfig


class TestAudioCapture:
    """Test suite for AudioCapture class"""
    
    @pytest.fixture
    def mock_sounddevice(self):
        """Mock sounddevice module"""
        with patch('whisper_transcriber.audio_capture.sd') as mock_sd:
            # Setup default device info
            mock_sd.query_devices.return_value = [
                {
                    'name': 'Built-in Microphone',
                    'max_input_channels': 2,
                    'default_samplerate': 44100.0,
                    'hostapi': 0
                },
                {
                    'name': 'USB Microphone',
                    'max_input_channels': 1,
                    'default_samplerate': 48000.0,
                    'hostapi': 0
                }
            ]
            mock_sd.default.device = (0, None)  # Default input device
            yield mock_sd
    
    @pytest.fixture
    def audio_capture(self, mock_sounddevice):
        """Create AudioCapture instance with mocked sounddevice"""
        return AudioCapture()
    
    def test_init_default_device(self, mock_sounddevice):
        """Test AudioCapture initialization with default device"""
        capture = AudioCapture()
        assert capture.device_id is None
        assert capture.stream is None
        assert capture.is_recording is False
        assert capture.audio_callback is None
    
    def test_init_specific_device(self, mock_sounddevice):
        """Test AudioCapture initialization with specific device"""
        capture = AudioCapture(device_id=1)
        assert capture.device_id == 1
    
    def test_list_devices(self, audio_capture, mock_sounddevice):
        """Test listing available audio devices"""
        devices = audio_capture.list_devices()
        
        assert len(devices) == 2
        assert devices[0].id == 0
        assert devices[0].name == "Built-in Microphone"
        assert devices[0].channels == 2
        assert devices[0].sample_rate == 44100
        assert devices[0].is_default is True
        
        assert devices[1].id == 1
        assert devices[1].name == "USB Microphone"
        assert devices[1].channels == 1
        assert devices[1].sample_rate == 48000
        assert devices[1].is_default is False
    
    def test_list_devices_no_input_channels(self, audio_capture, mock_sounddevice):
        """Test listing devices filters out devices with no input channels"""
        mock_sounddevice.query_devices.return_value = [
            {
                'name': 'Built-in Microphone',
                'max_input_channels': 2,
                'default_samplerate': 44100.0,
                'hostapi': 0
            },
            {
                'name': 'Output Only Device',
                'max_input_channels': 0,
                'default_samplerate': 48000.0,
                'hostapi': 0
            }
        ]
        
        devices = audio_capture.list_devices()
        assert len(devices) == 1
        assert devices[0].name == "Built-in Microphone"
    
    def test_start_recording_success(self, audio_capture, mock_sounddevice):
        """Test successful start of recording"""
        callback = Mock()
        mock_stream = MagicMock()
        mock_sounddevice.InputStream.return_value = mock_stream
        
        result = audio_capture.start_recording(callback)
        
        assert result is True
        assert audio_capture.is_recording is True
        assert audio_capture.audio_callback == callback
        assert audio_capture.stream == mock_stream
        
        # Verify stream configuration
        mock_sounddevice.InputStream.assert_called_once_with(
            device=None,
            channels=1,
            samplerate=16000,
            blocksize=1600,
            dtype='int16',
            callback=audio_capture._audio_callback
        )
        mock_stream.start.assert_called_once()
    
    def test_start_recording_with_device_id(self, mock_sounddevice):
        """Test start recording with specific device ID"""
        capture = AudioCapture(device_id=1)
        callback = Mock()
        mock_stream = MagicMock()
        mock_sounddevice.InputStream.return_value = mock_stream
        
        capture.start_recording(callback)
        
        mock_sounddevice.InputStream.assert_called_once_with(
            device=1,
            channels=1,
            samplerate=16000,
            blocksize=1600,
            dtype='int16',
            callback=capture._audio_callback
        )
    
    def test_start_recording_already_recording(self, audio_capture):
        """Test start recording when already recording"""
        audio_capture.is_recording = True
        callback = Mock()
        
        result = audio_capture.start_recording(callback)
        
        assert result is False
    
    def test_start_recording_error(self, audio_capture, mock_sounddevice):
        """Test start recording handles errors gracefully"""
        callback = Mock()
        mock_sounddevice.InputStream.side_effect = Exception("Device error")
        
        result = audio_capture.start_recording(callback)
        
        assert result is False
        assert audio_capture.is_recording is False
        assert audio_capture.stream is None
    
    def test_stop_recording_success(self, audio_capture, mock_sounddevice):
        """Test successful stop of recording"""
        # Setup recording state
        mock_stream = MagicMock()
        audio_capture.stream = mock_stream
        audio_capture.is_recording = True
        audio_capture.audio_callback = Mock()
        
        audio_capture.stop_recording()
        
        assert audio_capture.is_recording is False
        assert audio_capture.stream is None
        assert audio_capture.audio_callback is None
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
    
    def test_stop_recording_not_recording(self, audio_capture):
        """Test stop recording when not recording"""
        audio_capture.stop_recording()  # Should not raise any errors
        assert audio_capture.is_recording is False
    
    def test_audio_callback_processing(self, audio_capture):
        """Test audio callback processes data correctly"""
        # Setup
        user_callback = Mock()
        audio_capture.audio_callback = user_callback
        audio_capture.is_recording = True
        
        # Create test audio data
        test_data = np.array([[1000], [2000], [3000]], dtype=np.int16)
        
        # Call the internal callback
        audio_capture._audio_callback(test_data, frames=3, time=None, status=None)
        
        # Verify callback was called with byte data
        user_callback.assert_called_once()
        call_args = user_callback.call_args[0][0]
        assert isinstance(call_args, bytes)
        assert len(call_args) == 6  # 3 samples * 2 bytes per int16
    
    def test_audio_callback_not_recording(self, audio_capture):
        """Test audio callback does nothing when not recording"""
        user_callback = Mock()
        audio_capture.audio_callback = user_callback
        audio_capture.is_recording = False
        
        test_data = np.array([[1000]], dtype=np.int16)
        audio_capture._audio_callback(test_data, frames=1, time=None, status=None)
        
        user_callback.assert_not_called()
    
    def test_audio_callback_with_status_error(self, audio_capture, caplog):
        """Test audio callback handles status errors"""
        user_callback = Mock()
        audio_capture.audio_callback = user_callback
        audio_capture.is_recording = True
        
        test_data = np.array([[1000]], dtype=np.int16)
        # Create a mock status object that evaluates to True and has string representation
        status = MagicMock()
        status.__bool__.return_value = True
        status.__str__.return_value = "Buffer overflow"
        
        with caplog.at_level(logging.WARNING):
            audio_capture._audio_callback(test_data, frames=1, time=None, status=status)
        
        # Should still process data despite status error
        user_callback.assert_called_once()
        # Check that error was logged
        assert "Audio callback error" in caplog.text
    
    def test_get_current_device(self, audio_capture, mock_sounddevice):
        """Test getting current device information"""
        device = audio_capture.get_current_device()
        
        assert device is not None
        assert device.id == 0
        assert device.name == "Built-in Microphone"
        assert device.is_default is True
    
    def test_get_current_device_specific_id(self, mock_sounddevice):
        """Test getting current device with specific ID"""
        capture = AudioCapture(device_id=1)
        device = capture.get_current_device()
        
        assert device is not None
        assert device.id == 1
        assert device.name == "USB Microphone"
        assert device.is_default is False
    
    def test_set_device(self, audio_capture):
        """Test setting audio device"""
        audio_capture.set_device(1)
        assert audio_capture.device_id == 1
        
        audio_capture.set_device(None)
        assert audio_capture.device_id is None
    
    def test_set_device_while_recording(self, audio_capture):
        """Test setting device while recording fails"""
        audio_capture.is_recording = True
        
        with pytest.raises(RuntimeError, match="Cannot change device while recording"):
            audio_capture.set_device(1)
    
    def test_get_audio_config(self, audio_capture):
        """Test getting audio configuration"""
        config = audio_capture.get_audio_config()
        
        assert isinstance(config, AudioConfig)
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 1600
        assert config.format == "int16"
    
    def test_concurrent_start_stop(self, audio_capture, mock_sounddevice):
        """Test thread safety of start/stop operations"""
        callback = Mock()
        mock_stream = MagicMock()
        mock_sounddevice.InputStream.return_value = mock_stream
        
        def start_recording():
            audio_capture.start_recording(callback)
        
        def stop_recording():
            time.sleep(0.01)  # Small delay
            audio_capture.stop_recording()
        
        # Start threads
        start_thread = threading.Thread(target=start_recording)
        stop_thread = threading.Thread(target=stop_recording)
        
        start_thread.start()
        stop_thread.start()
        
        start_thread.join()
        stop_thread.join()
        
        # Should end in stopped state
        assert audio_capture.is_recording is False
        assert audio_capture.stream is None