import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys

# Mock tkinter before importing main
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.ttk'] = MagicMock()

from whisper_transcriber.main import WhisperTranscriberApp
from whisper_transcriber.models import InsertMethod


class TestEndToEnd:
    """End-to-end integration tests for the complete application flow"""
    
    @pytest.fixture
    def mock_all_dependencies(self):
        """Mock all external dependencies"""
        with patch('whisper_transcriber.main.ConfigManager') as mock_config:
            with patch('whisper_transcriber.main.AudioCapture') as mock_audio:
                with patch('whisper_transcriber.main.TranscriptionService') as mock_transcription:
                    with patch('whisper_transcriber.main.TextInserter') as mock_inserter:
                        with patch('whisper_transcriber.main.HotkeyManager') as mock_hotkey:
                            # Don't mock rumps - let the actual rumps library handle menu creation
                            yield {
                                'config': mock_config,
                                'audio': mock_audio,
                                'transcription': mock_transcription,
                                'inserter': mock_inserter,
                                'hotkey': mock_hotkey
                            }
    
    @pytest.mark.integration
    def test_complete_recording_flow(self, mock_all_dependencies):
        """Test complete flow from hotkey press to text insertion"""
        # Configure mocks
        config_mock = mock_all_dependencies['config'].return_value
        config_mock.get.side_effect = lambda key, default=None: {
            'hotkey': 'cmd+shift+r',
            'insertion_method': 'clipboard',
            'audio_device_id': None
        }.get(key, default)
        
        audio_mock = mock_all_dependencies['audio'].return_value
        transcription_mock = mock_all_dependencies['transcription'].return_value
        inserter_mock = mock_all_dependencies['inserter'].return_value
        hotkey_mock = mock_all_dependencies['hotkey'].return_value
        
        # Configure successful operations
        transcription_mock.start_server.return_value = True
        transcription_mock.is_server_running.return_value = True
        transcription_mock.connect_websocket.return_value = True
        audio_mock.start_recording.return_value = True
        
        # Create app
        app = WhisperTranscriberApp()
        
        # Simulate hotkey press to start recording
        hotkey_callback = hotkey_mock.register_hotkey.call_args[0][1]
        hotkey_callback()
        
        # Verify recording started
        assert app.is_recording is True
        transcription_mock.start_server.assert_called_once()
        transcription_mock.connect_websocket.assert_called_once()
        audio_mock.start_recording.assert_called_once()
        
        # Get audio callback
        audio_callback = audio_mock.start_recording.call_args[0][0]
        
        # Simulate audio data
        audio_data = b"test audio chunk"
        audio_callback(audio_data)
        
        # Verify audio sent to transcription
        transcription_mock.send_audio_chunk.assert_called_once_with(audio_data)
        
        # Simulate transcription result
        transcription_callback = transcription_mock.transcription_callback
        transcription_callback("Hello world", is_final=True)
        
        # Verify text inserted
        inserter_mock.insert_text.assert_called_once()
        call_args = inserter_mock.insert_text.call_args
        assert call_args[0][0] == "Hello world"
        assert call_args[0][1].value == "clipboard"
        
        # Simulate hotkey press to stop recording
        hotkey_callback()
        
        # Verify recording stopped
        assert app.is_recording is False
        audio_mock.stop_recording.assert_called_once()
        transcription_mock.disconnect_websocket.assert_called_once()
    
    @pytest.mark.integration
    def test_multiple_recording_sessions(self, mock_all_dependencies):
        """Test multiple start/stop recording sessions"""
        # Configure mocks
        config_mock = mock_all_dependencies['config'].return_value
        config_mock.get.return_value = False  # Disable notifications
        
        audio_mock = mock_all_dependencies['audio'].return_value
        transcription_mock = mock_all_dependencies['transcription'].return_value
        
        # All operations successful
        transcription_mock.start_server.return_value = True
        transcription_mock.is_server_running.return_value = True
        transcription_mock.connect_websocket.return_value = True
        audio_mock.start_recording.return_value = True
        
        # Create app
        app = WhisperTranscriberApp()
        
        # Run multiple sessions
        for session in range(3):
            # Start recording
            app.toggle_recording(None)
            assert app.is_recording is True
            
            # Stop recording
            app.toggle_recording(None)
            assert app.is_recording is False
        
        # Verify correct number of calls
        # Server is started once during init, then only checked if running
        assert transcription_mock.start_server.call_count >= 1
        assert transcription_mock.is_server_running.call_count == 3
        # Server is stopped on app quit, not during session stops
        assert transcription_mock.disconnect_websocket.call_count == 3
        assert audio_mock.start_recording.call_count == 3
        assert audio_mock.stop_recording.call_count == 3
    
    @pytest.mark.integration
    def test_concurrent_operations(self, mock_all_dependencies):
        """Test handling concurrent audio and transcription operations"""
        # Configure mocks
        config_mock = mock_all_dependencies['config'].return_value
        config_mock.get.side_effect = lambda key, default=None: {
            'hotkey': 'cmd+shift+r',
            'insertion_method': 'clipboard',
            'audio_device_id': None
        }.get(key, default)
        
        audio_mock = mock_all_dependencies['audio'].return_value
        transcription_mock = mock_all_dependencies['transcription'].return_value
        inserter_mock = mock_all_dependencies['inserter'].return_value
        
        # Setup successful operations
        transcription_mock.start_server.return_value = True
        transcription_mock.is_server_running.return_value = True
        transcription_mock.connect_websocket.return_value = True
        audio_mock.start_recording.return_value = True
        
        # Track operations
        audio_chunks = []
        transcriptions = []
        insertions = []
        
        def track_audio(chunk):
            audio_chunks.append(chunk)
            time.sleep(0.01)  # Simulate processing
        
        def track_insertion(text, method=None):
            insertions.append(text)
        
        transcription_mock.send_audio_chunk.side_effect = track_audio
        inserter_mock.insert_text.side_effect = track_insertion
        
        # Create app and start recording
        app = WhisperTranscriberApp()
        app.toggle_recording(None)
        
        # Get callbacks
        audio_callback = audio_mock.start_recording.call_args[0][0]
        transcription_callback = transcription_mock.transcription_callback
        
        # Simulate concurrent audio and transcription
        def send_audio():
            for i in range(5):
                audio_callback(f"chunk_{i}".encode())
                time.sleep(0.02)
        
        def send_transcriptions():
            time.sleep(0.01)  # Small offset
            for i in range(5):
                transcription_callback(f"Text {i}", is_final=True)
                time.sleep(0.02)
        
        # Run concurrently
        audio_thread = threading.Thread(target=send_audio)
        transcription_thread = threading.Thread(target=send_transcriptions)
        
        audio_thread.start()
        transcription_thread.start()
        
        audio_thread.join()
        transcription_thread.join()
        
        # Verify all operations completed
        assert len(audio_chunks) == 5
        assert len(insertions) == 5
        
        # Stop recording
        app.toggle_recording(None)
    
    @pytest.mark.integration
    def test_error_recovery_scenarios(self, mock_all_dependencies):
        """Test recovery from various error scenarios"""
        # Configure mocks
        config_mock = mock_all_dependencies['config'].return_value
        config_mock.get.return_value = False
        
        audio_mock = mock_all_dependencies['audio'].return_value
        transcription_mock = mock_all_dependencies['transcription'].return_value
        
        # Mock rumps.alert to prevent blocking dialogs
        with patch('whisper_transcriber.main.rumps.alert'):
            # Create app
            app = WhisperTranscriberApp()
            
            # Scenario 1: Server not running and start failure
            transcription_mock.is_server_running.return_value = False
            transcription_mock.start_server.return_value = False
            app.toggle_recording(None)
            assert app.is_recording is False
            
            # Scenario 2: WebSocket connection failure
            transcription_mock.is_server_running.return_value = True
            transcription_mock.start_server.return_value = True
            transcription_mock.connect_websocket.return_value = False
            app.toggle_recording(None)
            assert app.is_recording is False
            transcription_mock.stop_server.assert_called()
            
            # Scenario 3: Audio capture failure
            transcription_mock.connect_websocket.return_value = True
            audio_mock.start_recording.return_value = False
            app.toggle_recording(None)
            assert app.is_recording is False
            transcription_mock.disconnect_websocket.assert_called()
            
            # Scenario 4: Successful after failures
            audio_mock.start_recording.return_value = True
            app.toggle_recording(None)
            assert app.is_recording is True
    
    @pytest.mark.integration
    def test_preferences_integration(self, mock_all_dependencies):
        """Test preferences window integration"""
        # Configure mocks
        config_mock = mock_all_dependencies['config'].return_value
        config_mock.get.side_effect = lambda key, default=None: {
            'hotkey': 'cmd+shift+r',
            'model': 'tiny.en',
            'language': 'en',
            'insertion_method': 'auto',
            'audio_device': 'default',
            'audio_device_id': None,
            'start_at_login': False
        }.get(key, default)
        
        # Create app
        app = WhisperTranscriberApp()
        
        # Show preferences
        with patch('whisper_transcriber.main.SimplePreferencesWindow') as mock_prefs_class:
            mock_prefs = MagicMock()
            mock_prefs_class.return_value = mock_prefs
            
            app.show_preferences(None)
            
            # Verify preferences window created with correct components
            mock_prefs_class.assert_called_once_with(
                app.config_manager,
                app.audio_capture,
                app.hotkey_manager
            )
            mock_prefs.show.assert_called_once()