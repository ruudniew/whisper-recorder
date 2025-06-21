import pytest
import time
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import threading

from whisper_transcriber.audio_capture import AudioCapture
from whisper_transcriber.transcriber import TranscriptionService
from whisper_transcriber.models import ServerConfig, AudioConfig


class TestAudioToTranscription:
    """Integration tests for audio capture to transcription flow"""
    
    @pytest.fixture
    def server_config(self):
        """Create test server configuration"""
        return ServerConfig(
            host="localhost",
            port=9090,
            model="tiny.en",
            language="en"
        )
    
    @pytest.fixture
    def audio_config(self):
        """Create test audio configuration"""
        return AudioConfig()
    
    @pytest.mark.integration
    def test_audio_capture_to_transcription_flow(self, server_config, audio_config):
        """Test complete flow from audio capture to transcription"""
        # Setup components
        audio_capture = AudioCapture()
        transcription_service = TranscriptionService(server_config)
        
        # Track transcriptions
        transcriptions = []
        
        def handle_transcription(text, is_final):
            transcriptions.append((text, is_final))
        
        transcription_service.transcription_callback = handle_transcription
        
        # Mock server and WebSocket
        with patch.object(transcription_service, 'start_server', return_value=True):
            with patch.object(transcription_service, 'connect_websocket', return_value=True):
                with patch.object(transcription_service, 'send_audio_chunk') as mock_send:
                    with patch('whisper_transcriber.audio_capture.sd.InputStream'):
                        # Start recording
                        assert audio_capture.start_recording(
                            transcription_service.send_audio_chunk
                        )
                        
                        # Simulate audio callback
                        test_audio = np.array([[1000], [2000], [3000]], dtype=np.int16)
                        audio_capture._audio_callback(
                            test_audio, 
                            frames=3, 
                            time=None, 
                            status=None
                        )
                        
                        # Verify audio was sent
                        mock_send.assert_called_once()
                        audio_bytes = mock_send.call_args[0][0]
                        assert isinstance(audio_bytes, bytes)
                        assert len(audio_bytes) == 6  # 3 samples * 2 bytes
                        
                        # Stop recording
                        audio_capture.stop_recording()
    
    @pytest.mark.integration
    def test_multiple_audio_chunks_handling(self, server_config):
        """Test handling multiple audio chunks in sequence"""
        audio_capture = AudioCapture()
        transcription_service = TranscriptionService(server_config)
        
        chunks_sent = []
        
        # Mock send_audio_chunk to track calls
        def mock_send(chunk):
            chunks_sent.append(chunk)
        
        transcription_service.send_audio_chunk = mock_send
        
        with patch('whisper_transcriber.audio_capture.sd.InputStream'):
            audio_capture.start_recording(mock_send)
            
            # Send multiple chunks
            for i in range(5):
                test_audio = np.full((1024, 1), i * 100, dtype=np.int16)
                audio_capture._audio_callback(
                    test_audio,
                    frames=1024,
                    time=None,
                    status=None
                )
            
            # Verify all chunks were sent
            assert len(chunks_sent) == 5
            for chunk in chunks_sent:
                assert isinstance(chunk, bytes)
                assert len(chunk) == 2048  # 1024 samples * 2 bytes
            
            audio_capture.stop_recording()
    
    @pytest.mark.integration
    def test_error_recovery_in_audio_pipeline(self, server_config):
        """Test error recovery when audio processing fails"""
        audio_capture = AudioCapture()
        transcription_service = TranscriptionService(server_config)
        
        call_count = 0
        
        def failing_callback(chunk):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Simulated error")
        
        with patch('whisper_transcriber.audio_capture.sd.InputStream'):
            audio_capture.start_recording(failing_callback)
            
            # Send chunks - one should fail
            for i in range(3):
                test_audio = np.array([[i * 100]], dtype=np.int16)
                audio_capture._audio_callback(
                    test_audio,
                    frames=1,
                    time=None,
                    status=None
                )
            
            # Should have processed all chunks despite error
            assert call_count == 3
            
            # Audio capture should still be running
            assert audio_capture.is_recording
            
            audio_capture.stop_recording()
    
    @pytest.mark.integration
    @patch('whisper_transcriber.transcriber.websocket.WebSocketApp')
    def test_websocket_message_to_transcription(self, mock_ws_app, server_config):
        """Test WebSocket message processing to transcription callback"""
        transcription_service = TranscriptionService(server_config)
        
        received_transcriptions = []
        
        def handle_transcription(text, is_final):
            received_transcriptions.append((text, is_final))
        
        transcription_service.transcription_callback = handle_transcription
        
        # Setup WebSocket mock
        mock_ws = MagicMock()
        mock_ws_app.return_value = mock_ws
        
        # Connect WebSocket
        transcription_service.connect_websocket()
        
        # Get message handler
        on_message = mock_ws_app.call_args[1]['on_message']
        
        # Simulate incoming messages in WhisperLiveKit format
        messages = [
            '{"buffer_transcription": "Hello", "type": "transcription"}',
            '{"lines": [{"text": "Hello world"}], "type": "transcription"}',
            '{"buffer_transcription": "Testing", "type": "transcription"}',
            '{"lines": [{"text": "Testing transcription"}], "type": "transcription"}'
        ]
        
        for msg in messages:
            on_message(mock_ws, msg)
        
        # Verify all transcriptions received
        assert len(received_transcriptions) == 4
        assert received_transcriptions[0] == ("Hello", False)
        assert received_transcriptions[1] == ("Hello world", True)
        assert received_transcriptions[2] == ("Testing", False)
        assert received_transcriptions[3] == ("Testing transcription", True)
    
    @pytest.mark.integration
    def test_concurrent_audio_capture_and_sending(self, server_config):
        """Test concurrent audio capture and sending"""
        audio_capture = AudioCapture()
        transcription_service = TranscriptionService(server_config)
        
        chunks_received = []
        lock = threading.Lock()
        
        def thread_safe_send(chunk):
            with lock:
                chunks_received.append(chunk)
                time.sleep(0.01)  # Simulate processing delay
        
        with patch('whisper_transcriber.audio_capture.sd.InputStream'):
            audio_capture.start_recording(thread_safe_send)
            
            # Start multiple threads sending audio
            threads = []
            
            def send_audio_batch(thread_id):
                for i in range(5):
                    test_audio = np.array([[thread_id * 1000 + i]], dtype=np.int16)
                    audio_capture._audio_callback(
                        test_audio,
                        frames=1,
                        time=None,
                        status=None
                    )
                    time.sleep(0.005)
            
            # Start threads
            for i in range(3):
                t = threading.Thread(target=send_audio_batch, args=(i,))
                threads.append(t)
                t.start()
            
            # Wait for completion
            for t in threads:
                t.join()
            
            # Verify all chunks received
            assert len(chunks_received) == 15  # 3 threads * 5 chunks each
            
            audio_capture.stop_recording()