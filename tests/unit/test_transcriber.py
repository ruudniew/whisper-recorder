import asyncio
import json
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
import subprocess
import threading
import time
import websocket

from whisper_transcriber.transcriber import TranscriptionService, TranscriptionError
from whisper_transcriber.models import ServerConfig


class TestTranscriptionService:
    """Test suite for TranscriptionService class"""
    
    @pytest.fixture
    def server_config(self):
        """Create test server configuration"""
        return ServerConfig(
            host="localhost",
            port=9090,
            model="tiny.en",
            language="en",
            vad_enabled=False,
            use_gpu=False
        )
    
    @pytest.fixture
    def transcription_service(self, server_config):
        """Create TranscriptionService instance"""
        return TranscriptionService(server_config)
    
    def test_init(self, transcription_service, server_config):
        """Test TranscriptionService initialization"""
        assert transcription_service.server_config == server_config
        assert transcription_service.server_process is None
        assert transcription_service.websocket_client is None
        assert transcription_service.is_connected is False
        assert transcription_service.transcription_callback is None
    
    @patch('whisper_transcriber.transcriber.shutil.which')
    @patch('whisper_transcriber.transcriber.subprocess.Popen')
    def test_start_server_success(self, mock_popen, mock_which, transcription_service):
        """Test successful server startup"""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        # Mock finding whisperlivekit-server
        mock_which.return_value = '/usr/local/bin/whisperlivekit-server'
        
        result = transcription_service.start_server()
        
        assert result is True
        assert transcription_service.server_process == mock_process
        
        # Verify correct command was used
        expected_cmd = [
            '/usr/local/bin/whisperlivekit-server',
            '--host', 'localhost',
            '--port', '9090',
            '--model', 'tiny.en',
            '--lan', 'en',  # Note: changed from --language to --lan
            '--no-vad',  # VAD is disabled by default now
            '--raw-pcm'  # Enable raw PCM mode
        ]
        mock_popen.assert_called_once()
        actual_cmd = mock_popen.call_args[0][0]
        assert actual_cmd == expected_cmd
    
    @patch('whisper_transcriber.transcriber.shutil.which')
    @patch('whisper_transcriber.transcriber.subprocess.Popen')
    def test_start_server_with_gpu(self, mock_popen, mock_which, server_config):
        """Test server startup with GPU enabled"""
        server_config = ServerConfig(use_gpu=True)
        service = TranscriptionService(server_config)
        
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        # Mock finding whisperlivekit-server
        mock_which.return_value = '/usr/local/bin/whisperlivekit-server'
        
        service.start_server()
        
    
    @patch('whisper_transcriber.transcriber.subprocess.Popen')
    def test_start_server_already_running(self, mock_popen, transcription_service):
        """Test starting server when already running"""
        # Set up existing process
        transcription_service.server_process = MagicMock()
        transcription_service.server_process.poll.return_value = None
        
        result = transcription_service.start_server()
        
        assert result is True  # Already running
        mock_popen.assert_not_called()  # Should not start new process
    
    @patch('whisper_transcriber.transcriber.subprocess.Popen')
    def test_start_server_failure(self, mock_popen, transcription_service):
        """Test server startup failure"""
        mock_popen.side_effect = Exception("Failed to start")
        
        result = transcription_service.start_server()
        
        assert result is False
        assert transcription_service.server_process is None
    
    @patch('whisper_transcriber.transcriber.threading.Thread')
    @patch('whisper_transcriber.transcriber.websocket.WebSocketApp')
    def test_connect_websocket_success(self, mock_websocket_app, mock_thread, transcription_service):
        """Test successful WebSocket connection"""
        # Setup mock WebSocket
        mock_ws = MagicMock()
        mock_websocket_app.return_value = mock_ws
        
        # Setup mock thread
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        # Set callback
        callback = Mock()
        transcription_service.transcription_callback = callback
        
        # Simulate successful connection by setting is_connected before the timeout
        def side_effect(*args, **kwargs):
            # After creating thread, simulate connection
            if mock_thread_instance.start.called:
                transcription_service.is_connected = True
        
        mock_thread_instance.start.side_effect = side_effect
        
        result = transcription_service.connect_websocket()
        
        # Verify WebSocket created with correct URL
        mock_websocket_app.assert_called_once()
        assert mock_websocket_app.call_args[0][0] == "ws://localhost:9090/asr"
        
        # Verify thread was started
        mock_thread_instance.start.assert_called_once()
        
        assert result is True
    
    @patch('whisper_transcriber.transcriber.websocket.WebSocketApp')
    def test_connect_websocket_no_callback(self, mock_websocket_app, transcription_service):
        """Test WebSocket connection fails without callback"""
        result = transcription_service.connect_websocket()
        
        assert result is False
        mock_websocket_app.assert_not_called()
    
    def test_send_audio_chunk_not_connected(self, transcription_service):
        """Test sending audio when not connected"""
        audio_data = b"test audio data"
        
        # Should not raise error
        transcription_service.send_audio_chunk(audio_data)
    
    def test_send_audio_chunk_connected(self, transcription_service):
        """Test sending audio chunk when connected"""
        # Setup websocket mock
        mock_ws = MagicMock()
        transcription_service.websocket_client = mock_ws
        transcription_service.is_connected = True
        
        audio_data = b"test audio data"
        transcription_service.send_audio_chunk(audio_data)
        
        # Verify data was sent directly as raw PCM
        mock_ws.send.assert_called_once_with(audio_data, opcode=websocket.ABNF.OPCODE_BINARY)
    
    def test_handle_transcription_final(self, transcription_service):
        """Test handling final transcription result"""
        callback = Mock()
        transcription_service.transcription_callback = callback
        
        text = "Hello world"
        transcription_service.handle_transcription(text, is_final=True)
        
        callback.assert_called_once_with(text, True)
    
    def test_handle_transcription_partial(self, transcription_service):
        """Test handling partial transcription result"""
        callback = Mock()
        transcription_service.transcription_callback = callback
        
        text = "Hello"
        transcription_service.handle_transcription(text, is_final=False)
        
        callback.assert_called_once_with(text, False)
    
    def test_handle_transcription_no_callback(self, transcription_service):
        """Test handling transcription without callback"""
        # Should not raise error
        transcription_service.handle_transcription("test", is_final=True)
    
    def test_stop_server(self, transcription_service):
        """Test stopping server"""
        # Setup mock process
        mock_process = MagicMock()
        transcription_service.server_process = mock_process
        
        transcription_service.stop_server()
        
        # Verify process was terminated
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
        assert transcription_service.server_process is None
    
    def test_stop_server_timeout(self, transcription_service):
        """Test stopping server with timeout"""
        # Setup mock process that times out
        mock_process = MagicMock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
        transcription_service.server_process = mock_process
        
        transcription_service.stop_server()
        
        # Should kill after timeout
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
    
    def test_stop_server_not_running(self, transcription_service):
        """Test stopping server when not running"""
        # Should not raise error
        transcription_service.stop_server()
    
    def test_disconnect_websocket(self, transcription_service):
        """Test disconnecting WebSocket"""
        # Setup mock WebSocket
        mock_ws = MagicMock()
        transcription_service.websocket_client = mock_ws
        transcription_service.is_connected = True
        
        transcription_service.disconnect_websocket()
        
        # Verify disconnection
        mock_ws.close.assert_called_once()
        assert transcription_service.is_connected is False
        assert transcription_service.websocket_client is None
    
    def test_disconnect_websocket_not_connected(self, transcription_service):
        """Test disconnecting when not connected"""
        # Should not raise error
        transcription_service.disconnect_websocket()
    
    @patch('whisper_transcriber.transcriber.websocket.WebSocketApp')
    def test_websocket_on_message(self, mock_websocket_app, transcription_service):
        """Test WebSocket message handling"""
        # Setup
        callback = Mock()
        transcription_service.transcription_callback = callback
        
        # Create WebSocket (to get callbacks)
        transcription_service.connect_websocket()
        on_message = mock_websocket_app.call_args[1]['on_message']
        
        # Test message with transcription
        message = json.dumps({
            "lines": [{"text": "Hello world"}],
            "type": "transcription"
        })
        on_message(None, message)
        
        # Verify callback called
        callback.assert_called_with("Hello world", True)
    
    @patch('whisper_transcriber.transcriber.websocket.WebSocketApp')
    def test_websocket_on_error(self, mock_websocket_app, transcription_service):
        """Test WebSocket error handling"""
        # Setup
        transcription_service.transcription_callback = Mock()
        transcription_service.connect_websocket()
        on_error = mock_websocket_app.call_args[1]['on_error']
        
        # Simulate error
        error = Exception("Connection error")
        on_error(None, error)
        
        # Should handle error gracefully
        assert transcription_service.is_connected is False
    
    @patch('whisper_transcriber.transcriber.websocket.WebSocketApp')
    def test_websocket_on_close(self, mock_websocket_app, transcription_service):
        """Test WebSocket close handling"""
        # Setup
        transcription_service.transcription_callback = Mock()
        transcription_service.is_connected = True
        transcription_service.connect_websocket()
        on_close = mock_websocket_app.call_args[1]['on_close']
        
        # Simulate close
        on_close(None, 1000, "Normal closure")
        
        # Verify state updated
        assert transcription_service.is_connected is False
    
    def test_is_server_running(self, transcription_service):
        """Test checking if server is running"""
        # No process
        assert transcription_service.is_server_running() is False
        
        # Process running
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        transcription_service.server_process = mock_process
        assert transcription_service.is_server_running() is True
        
        # Process terminated
        mock_process.poll.return_value = 0
        assert transcription_service.is_server_running() is False
    
    def test_get_server_info(self, transcription_service):
        """Test getting server information"""
        info = transcription_service.get_server_info()
        
        assert info["host"] == "localhost"
        assert info["port"] == 9090
        assert info["model"] == "tiny.en"
        assert info["language"] == "en"
        assert info["websocket_url"] == "ws://localhost:9090/asr"
        assert info["is_running"] is False
        assert info["is_connected"] is False
    
    def test_restart_server(self, transcription_service):
        """Test restarting server"""
        with patch.object(transcription_service, 'stop_server') as mock_stop:
            with patch.object(transcription_service, 'start_server') as mock_start:
                mock_start.return_value = True
                
                result = transcription_service.restart_server()
                
                mock_stop.assert_called_once()
                mock_start.assert_called_once()
                assert result is True
    
    @patch('whisper_transcriber.transcriber.websocket.WebSocketApp')
    def test_incremental_buffer_transcription(self, mock_websocket_app, transcription_service):
        """Test that incremental buffer transcriptions only send new text"""
        # Setup
        transcription_service.transcription_callback = Mock()
        transcription_service.is_connected = True
        transcription_service.connect_websocket()
        on_message = mock_websocket_app.call_args[1]['on_message']
        
        # Simulate incremental buffer updates like WhisperLiveKit sends
        messages = [
            {"buffer_transcription": "Hello"},
            {"buffer_transcription": "Hello world"},
            {"buffer_transcription": "Hello world, how"},
            {"buffer_transcription": "Hello world, how are you"},
            {"buffer_transcription": "Hello world, how are you today?"}
        ]
        
        expected_calls = []
        
        # Process each message
        for i, msg in enumerate(messages):
            on_message(None, json.dumps(msg))
            
        # Verify only new text was sent each time
        calls = transcription_service.transcription_callback.call_args_list
        assert len(calls) == 5
        
        # First message sends full text
        assert calls[0][0][0] == "Hello"
        assert calls[0][0][1] is False  # is_final
        
        # Subsequent messages only send new portions
        assert calls[1][0][0] == " world"
        assert calls[2][0][0] == ", how"
        assert calls[3][0][0] == " are you"
        assert calls[4][0][0] == " today?"
    
    @patch('whisper_transcriber.transcriber.websocket.WebSocketApp')
    def test_duplicate_line_prevention(self, mock_websocket_app, transcription_service):
        """Test that duplicate lines are not sent multiple times"""
        # Setup
        transcription_service.transcription_callback = Mock()
        transcription_service.is_connected = True
        transcription_service.connect_websocket()
        on_message = mock_websocket_app.call_args[1]['on_message']
        
        # Simulate messages with duplicate lines
        messages = [
            {"lines": [{"text": "First line"}]},
            {"lines": [{"text": "First line"}, {"text": "Second line"}]},
            {"lines": [{"text": "First line"}, {"text": "Second line"}]},  # Duplicate
            {"lines": [{"text": "First line"}, {"text": "Second line"}, {"text": "Third line"}]}
        ]
        
        # Process messages
        for msg in messages:
            on_message(None, json.dumps(msg))
        
        # Verify each unique line was only sent once
        calls = transcription_service.transcription_callback.call_args_list
        assert len(calls) == 3  # Only 3 unique lines
        
        assert calls[0][0][0] == "First line"
        assert calls[0][0][1] is True  # is_final
        assert calls[1][0][0] == "Second line"
        assert calls[2][0][0] == "Third line"
    
    @patch('whisper_transcriber.transcriber.websocket.WebSocketApp')
    def test_transcription_state_reset_on_disconnect(self, mock_websocket_app, transcription_service):
        """Test that transcription tracking state is reset when disconnecting"""
        # Setup
        transcription_service.transcription_callback = Mock()
        transcription_service.is_connected = True
        transcription_service.websocket_client = mock_websocket_app.return_value
        
        # Simulate some transcription activity to populate tracking variables
        transcription_service._sent_texts = {"test1", "test2"}
        transcription_service._last_buffer_text = "Some buffer text"
        
        # Disconnect
        transcription_service.disconnect_websocket()
        
        # Verify tracking state was reset
        assert len(transcription_service._sent_texts) == 0
        assert transcription_service._last_buffer_text == ""