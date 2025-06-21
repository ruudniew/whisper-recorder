import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from whisper_transcriber.transcriber import TranscriptionService
from whisper_transcriber.text_inserter import TextInserter
from whisper_transcriber.models import ServerConfig, InsertMethod


class TestTranscriptionToInsertion:
    """Integration tests for transcription to text insertion flow"""
    
    @pytest.fixture
    def server_config(self):
        """Create test server configuration"""
        return ServerConfig()
    
    @pytest.fixture
    def transcription_service(self, server_config):
        """Create transcription service instance"""
        return TranscriptionService(server_config)
    
    @pytest.fixture
    def text_inserter(self):
        """Create text inserter instance"""
        return TextInserter()
    
    @pytest.mark.integration
    @patch('whisper_transcriber.text_inserter.pyperclip')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_transcription_to_clipboard_insertion(self, mock_controller_class, 
                                                 mock_pyperclip, 
                                                 transcription_service, 
                                                 text_inserter):
        """Test transcription results are inserted via clipboard"""
        # Setup mocks
        mock_pyperclip.paste.return_value = "original"
        mock_controller = MagicMock()
        mock_controller_class.return_value = mock_controller
        
        # Track insertions
        insertions = []
        
        def handle_transcription(text, is_final):
            if is_final:
                text_inserter.insert_text(text, InsertMethod.CLIPBOARD)
                insertions.append(text)
        
        transcription_service.transcription_callback = handle_transcription
        
        # Simulate transcriptions
        transcription_service.handle_transcription("Hello", False)
        transcription_service.handle_transcription("Hello world", True)
        transcription_service.handle_transcription("Testing", False)
        transcription_service.handle_transcription("Testing insertion", True)
        
        # Verify only final transcriptions were inserted
        assert len(insertions) == 2
        assert insertions[0] == "Hello world"
        assert insertions[1] == "Testing insertion"
        
        # Verify clipboard operations
        assert mock_pyperclip.copy.call_count == 4  # 2 insertions * (copy text + restore)
        assert mock_controller_class.call_count == 2  # Controller created for each insertion
    
    @pytest.mark.integration
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_transcription_to_keyboard_insertion(self, mock_controller_class,
                                                transcription_service,
                                                text_inserter):
        """Test transcription results are inserted via keyboard"""
        # Setup mock
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        # Track typed text
        typed_texts = []
        
        def track_typing(text):
            typed_texts.append(text)
        
        mock_controller.type.side_effect = track_typing
        
        def handle_transcription(text, is_final):
            if is_final:
                text_inserter.insert_text(text, InsertMethod.KEYBOARD)
        
        transcription_service.transcription_callback = handle_transcription
        
        # Simulate transcriptions
        transcription_service.handle_transcription("Short", True)
        transcription_service.handle_transcription("Another text", True)
        
        # Verify keyboard typing
        assert len(typed_texts) == 2
        assert typed_texts[0] == "Short"
        assert typed_texts[1] == "Another text"
    
    @pytest.mark.integration
    @patch('whisper_transcriber.text_inserter.pyperclip')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_auto_method_selection(self, mock_controller_class, mock_pyperclip,
                                  transcription_service, text_inserter):
        """Test automatic method selection based on text characteristics"""
        # Setup mocks
        mock_pyperclip.paste.return_value = ""
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        # Mock the pressed context manager
        mock_pressed = MagicMock()
        mock_pressed.__enter__ = Mock(return_value=None)
        mock_pressed.__exit__ = Mock(return_value=None)
        mock_controller.pressed.return_value = mock_pressed
        
        clipboard_texts = []
        keyboard_texts = []
        
        # Track which method was used
        def track_clipboard(text):
            if text != "":  # Ignore clipboard restore
                clipboard_texts.append(text)
        
        def track_keyboard(text):
            keyboard_texts.append(text)
        
        mock_pyperclip.copy.side_effect = track_clipboard
        mock_controller.type.side_effect = track_keyboard
        
        def handle_transcription(text, is_final):
            if is_final:
                text_inserter.insert_text(text, InsertMethod.AUTO)
        
        transcription_service.transcription_callback = handle_transcription
        
        # Test short text (should use keyboard)
        transcription_service.handle_transcription("Hi", True)
        
        # Test long text (should use clipboard)
        long_text = "This is a very long transcription that exceeds the threshold for keyboard typing"
        transcription_service.handle_transcription(long_text, True)
        
        # Test text with newlines (should use clipboard)
        multiline_text = "Line 1\nLine 2\nLine 3"
        transcription_service.handle_transcription(multiline_text, True)
        
        # Verify method selection
        assert len(keyboard_texts) == 1
        assert keyboard_texts[0] == "Hi"
        
        assert len(clipboard_texts) == 2
        assert long_text in clipboard_texts
        assert multiline_text in clipboard_texts
    
    @pytest.mark.integration
    def test_rapid_transcription_handling(self, transcription_service, text_inserter):
        """Test handling rapid succession of transcriptions"""
        with patch.object(text_inserter, 'insert_text') as mock_insert:
            insertions = []
            
            def track_insertion(text, method=None):
                insertions.append((text, time.time()))
            
            mock_insert.side_effect = track_insertion
            
            def handle_transcription(text, is_final):
                if is_final:
                    text_inserter.insert_text(text)
            
            transcription_service.transcription_callback = handle_transcription
            
            # Simulate rapid transcriptions
            start_time = time.time()
            for i in range(10):
                transcription_service.handle_transcription(f"Text {i}", True)
                time.sleep(0.01)  # Small delay
            
            # Verify all transcriptions were handled
            assert len(insertions) == 10
            
            # Verify they were processed in order
            for i in range(10):
                assert insertions[i][0] == f"Text {i}"
            
            # Verify timing (should complete quickly)
            total_time = insertions[-1][1] - start_time
            assert total_time < 1.0  # Should complete in under 1 second
    
    @pytest.mark.integration
    def test_special_characters_handling(self, transcription_service, text_inserter):
        """Test handling of special characters in transcriptions"""
        with patch.object(text_inserter, 'insert_text') as mock_insert:
            special_texts = []
            
            def track_special(text, method=None):
                special_texts.append(text)
            
            mock_insert.side_effect = track_special
            
            def handle_transcription(text, is_final):
                if is_final:
                    text_inserter.insert_text(text)
            
            transcription_service.transcription_callback = handle_transcription
            
            # Test various special characters
            test_texts = [
                "Hello! How are you?",
                "Price: $19.99 (on sale)",
                "Email: test@example.com",
                "Math: 2+2=4, 3*3=9",
                "Symbols: @#$%^&*()",
                "Unicode: café, naïve, 你好"
            ]
            
            for text in test_texts:
                transcription_service.handle_transcription(text, True)
            
            # Verify all special characters preserved
            assert len(special_texts) == len(test_texts)
            for i, text in enumerate(test_texts):
                assert special_texts[i] == text
    
    @pytest.mark.integration
    def test_error_recovery_in_insertion(self, transcription_service, text_inserter):
        """Test error recovery when text insertion fails"""
        with patch.object(text_inserter, '_clipboard_method') as mock_clipboard:
            with patch.object(text_inserter, '_keyboard_method') as mock_keyboard:
                # Make clipboard method fail
                mock_clipboard.side_effect = Exception("Clipboard error")
                
                successful_insertions = []
                
                def track_keyboard(text):
                    successful_insertions.append(text)
                
                mock_keyboard.side_effect = track_keyboard
                
                def handle_transcription(text, is_final):
                    if is_final:
                        text_inserter.insert_text(text, InsertMethod.CLIPBOARD)
                
                transcription_service.transcription_callback = handle_transcription
                
                # Try to insert text (should fallback to keyboard)
                transcription_service.handle_transcription("Fallback test", True)
                
                # Verify fallback worked
                assert len(successful_insertions) == 1
                assert successful_insertions[0] == "Fallback test"