import pytest
from unittest.mock import Mock, MagicMock, patch, call
import os
import sys

# Mock UI libraries before importing main
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.ttk'] = MagicMock()

# Create a proper mock for rumps
mock_rumps = MagicMock()
# Mock necessary rumps functions
mock_rumps.notification = MagicMock()

# Mock MenuItem class
class MockMenuItem:
    def __init__(self, title, callback=None):
        self.title = title
        self.callback = callback

mock_rumps.MenuItem = MockMenuItem
mock_rumps.separator = MagicMock()  # Just a separator marker

# Make rumps.App return a class that has proper attributes
class MockApp:
    def __init__(self, *args, **kwargs):
        self.title = kwargs.get('title', '')
        self.icon = kwargs.get('icon', None)
        self.menu = []
        self.is_recording = False
        
    def __setattr__(self, name, value):
        # Allow setting attributes normally
        object.__setattr__(self, name, value)
    
    def run(self):
        # Mock run method
        pass
        
mock_rumps.App = MockApp
sys.modules['rumps'] = mock_rumps

from whisper_transcriber.main import WhisperTranscriberApp, main


class TestWhisperTranscriberApp:
    """Test suite for WhisperTranscriberApp class"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies"""
        with patch('whisper_transcriber.main.ConfigManager') as mock_config_manager:
            with patch('whisper_transcriber.main.AudioCapture') as mock_audio_capture:
                with patch('whisper_transcriber.main.TranscriptionService') as mock_transcription_service:
                    with patch('whisper_transcriber.main.TextInserter') as mock_text_inserter:
                        with patch('whisper_transcriber.main.HotkeyManager') as mock_hotkey_manager:
                            yield {
                                'config_manager': mock_config_manager,
                                'audio_capture': mock_audio_capture,
                                'transcription_service': mock_transcription_service,
                                'text_inserter': mock_text_inserter,
                                'hotkey_manager': mock_hotkey_manager
                            }
    
    @pytest.fixture
    def app(self, mock_dependencies):
        """Create WhisperTranscriberApp instance with mocked dependencies"""
        # Configure mocks
        mock_config = mock_dependencies['config_manager'].return_value
        mock_config.get.side_effect = lambda key, default=None: {
            'hotkey': 'cmd+shift+r',
            'audio_device_id': None
        }.get(key, default)
        
        # Create app
        app = WhisperTranscriberApp()
        return app
    
    def test_init(self, app, mock_dependencies):
        """Test app initialization"""
        assert app.is_recording is False
        
        # Verify components initialized
        mock_dependencies['config_manager'].assert_called_once()
        mock_dependencies['audio_capture'].assert_called_once()
        mock_dependencies['transcription_service'].assert_called_once()
        mock_dependencies['text_inserter'].assert_called_once()
        mock_dependencies['hotkey_manager'].assert_called_once()
    
    def test_toggle_recording_start(self, app):
        """Test starting recording"""
        # Mock successful start
        app.transcription_service.is_server_running.return_value = True
        app.transcription_service.connect_websocket.return_value = True
        app.audio_capture.start_recording.return_value = True
        
        # Start recording
        app.toggle_recording(None)
        
        # Verify recording started
        assert app.is_recording is True
        app.transcription_service.is_server_running.assert_called_once()
        app.transcription_service.connect_websocket.assert_called_once()
        app.audio_capture.start_recording.assert_called_once()
    
    def test_toggle_recording_stop(self, app):
        """Test stopping recording"""
        # Set recording state
        app.is_recording = True
        
        # Stop recording
        app.toggle_recording(None)
        
        # Verify recording stopped
        assert app.is_recording is False
        app.audio_capture.stop_recording.assert_called_once()
        app.transcription_service.disconnect_websocket.assert_called_once()
    
    def test_toggle_recording_server_start_failure(self, app):
        """Test handling server start failure"""
        # Server is not running and fails to start
        app.transcription_service.is_server_running.return_value = False
        app.transcription_service.start_server.return_value = False
        
        # Mock rumps.alert to prevent blocking dialog
        with patch('whisper_transcriber.main.rumps.alert'):
            # Try to start recording
            app.toggle_recording(None)
        
        # Should not start recording
        assert app.is_recording is False
        app.audio_capture.start_recording.assert_not_called()
        app.transcription_service.is_server_running.assert_called_once()
        # start_server is called twice: once on init, once in toggle_recording
        assert app.transcription_service.start_server.call_count == 2
    
    def test_toggle_recording_websocket_failure(self, app):
        """Test handling WebSocket connection failure"""
        app.transcription_service.is_server_running.return_value = True
        app.transcription_service.connect_websocket.return_value = False
        
        # Mock rumps.alert to prevent blocking dialog
        with patch('whisper_transcriber.main.rumps.alert'):
            # Try to start recording
            app.toggle_recording(None)
        
        # Should not start recording
        assert app.is_recording is False
        app.audio_capture.start_recording.assert_not_called()
        app.transcription_service.stop_server.assert_called_once()
    
    def test_toggle_recording_audio_failure(self, app):
        """Test handling audio capture failure"""
        app.transcription_service.is_server_running.return_value = True
        app.transcription_service.connect_websocket.return_value = True
        app.audio_capture.start_recording.return_value = False
        
        # Mock rumps.alert to prevent blocking dialog
        with patch('whisper_transcriber.main.rumps.alert'):
            # Try to start recording
            app.toggle_recording(None)
        
        # Should clean up
        assert app.is_recording is False
        app.transcription_service.disconnect_websocket.assert_called_once()
        app.transcription_service.stop_server.assert_called_once()
    
    def test_handle_audio_chunk(self, app):
        """Test audio chunk handling"""
        audio_data = b"test audio data"
        
        app._handle_audio_chunk(audio_data)
        
        app.transcription_service.send_audio_chunk.assert_called_once_with(audio_data)
    
    def test_handle_transcription_final(self, app):
        """Test handling final transcription"""
        text = "Hello world"
        
        # Configure insertion method mock
        app.config_manager.get.side_effect = lambda key, default=None: {
            'insertion_method': 'clipboard'
        }.get(key, default)
        
        app._handle_transcription(text, is_final=True)
        
        # Check that insert_text was called with text and method
        app.text_inserter.insert_text.assert_called_once()
        call_args = app.text_inserter.insert_text.call_args
        assert call_args[0][0] == text  # First positional arg
        assert call_args[0][1].value == 'clipboard'  # Second arg is InsertMethod enum
    
    def test_handle_transcription_partial(self, app):
        """Test handling partial transcription (ignored)"""
        text = "Hello"
        
        app._handle_transcription(text, is_final=False)
        
        # Partial transcriptions should be ignored
        app.text_inserter.insert_text.assert_not_called()
    
    def test_show_preferences(self, app):
        """Test showing preferences window"""
        with patch('whisper_transcriber.main.SimplePreferencesWindow') as mock_prefs:
            app.show_preferences(None)
            
            mock_prefs.assert_called_once_with(
                app.config_manager,
                app.audio_capture,
                app.hotkey_manager
            )
            mock_prefs.return_value.show.assert_called_once()
    
    @patch('whisper_transcriber.main.rumps.quit_application')
    def test_quit_application(self, mock_quit, app):
        """Test quitting application"""
        # Set recording state
        app.is_recording = True
        
        # Quit
        app.quit_application(None)
        
        # Verify cleanup
        app.audio_capture.stop_recording.assert_called_once()
        app.transcription_service.stop_server.assert_called_once()
        app.hotkey_manager.stop_listening.assert_called_once()
        mock_quit.assert_called_once()
    
    def test_menu_setup(self, app):
        """Test menu items are properly set up"""
        # Get menu items (menu is a list) and convert titles to strings
        menu_titles = []
        for item in app.menu:
            if hasattr(item, 'title'):
                try:
                    # Handle both mock and real rumps objects
                    if callable(item.title):
                        # If title is a method (real rumps)
                        title = str(item.title())
                    else:
                        # If title is a property (mock)
                        title = str(item.title)
                    # Only add if it's a real menu item title
                    if title and not title.startswith('<MagicMock'):
                        menu_titles.append(title)
                except:
                    # Skip any problematic items
                    pass
        
        # Verify expected menu items
        assert any("Recording" in title for title in menu_titles), f"No Recording item found in {menu_titles}"
        assert "Preferences..." in menu_titles
        assert "About" in menu_titles
        assert "Quit" in menu_titles
    
    def test_hotkey_registration(self, app):
        """Test hotkey is registered on startup"""
        app.hotkey_manager.register_hotkey.assert_called_once_with(
            'cmd+shift+r',
            app.toggle_recording_hotkey
        )
        app.hotkey_manager.start_listening.assert_called_once()
    
    def test_toggle_recording_hotkey(self, app):
        """Test hotkey triggers recording toggle"""
        app.toggle_recording = Mock()
        
        app.toggle_recording_hotkey()
        
        app.toggle_recording.assert_called_once_with(None)
    


class TestMain:
    """Test suite for main entry point"""
    
    @patch('whisper_transcriber.main.WhisperTranscriberApp')
    def test_main_function(self, mock_app_class):
        """Test main function creates and runs app"""
        mock_app = MagicMock()
        mock_app_class.return_value = mock_app
        
        # Ensure run() doesn't actually run
        mock_app.run = MagicMock()
        
        main()
        
        mock_app_class.assert_called_once()
        mock_app.run.assert_called_once()