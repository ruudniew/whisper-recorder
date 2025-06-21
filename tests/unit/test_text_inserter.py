import time
import pytest
from unittest.mock import Mock, patch, call

from whisper_transcriber.text_inserter import TextInserter
from whisper_transcriber.models import InsertMethod


class TestTextInserter:
    """Test suite for TextInserter class"""
    
    @pytest.fixture
    def text_inserter(self):
        """Create TextInserter instance"""
        return TextInserter()
    
    def test_init(self, text_inserter):
        """Test TextInserter initialization"""
        assert text_inserter.original_clipboard is None
    
    @patch('whisper_transcriber.text_inserter.pyperclip')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_insert_text_clipboard_method(self, mock_controller_class, mock_pyperclip, text_inserter):
        """Test text insertion using clipboard method"""
        # Setup mocks
        mock_pyperclip.paste.return_value = "original content"
        
        # Insert text
        text_inserter.insert_text("Hello World", method=InsertMethod.CLIPBOARD)
        
        # Verify clipboard operations
        mock_pyperclip.paste.assert_called_once()  # Save original
        mock_pyperclip.copy.assert_any_call("Hello World")  # Copy new text
        mock_pyperclip.copy.assert_any_call("original content")  # Restore original
        
        # Verify keyboard controller was used
        mock_controller_class.assert_called()  # Controller was instantiated
    
    @patch('whisper_transcriber.text_inserter.pyperclip')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_insert_text_keyboard_method(self, mock_controller_class, mock_pyperclip, text_inserter):
        """Test text insertion using keyboard method"""
        # Setup mock controller
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        # Insert text
        text_inserter.insert_text("Hello", method=InsertMethod.KEYBOARD)
        
        # Verify keyboard typing
        mock_controller.type.assert_called_once_with("Hello")
        
        # Verify clipboard not used
        mock_pyperclip.paste.assert_not_called()
        mock_pyperclip.copy.assert_not_called()
    
    @patch('whisper_transcriber.text_inserter.pyperclip')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_insert_text_auto_method_short(self, mock_controller_class, mock_pyperclip, text_inserter):
        """Test auto method chooses keyboard for short text"""
        # Setup mock controller
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        # Insert short text
        text_inserter.insert_text("Hi", method=InsertMethod.AUTO)
        
        # Should use keyboard method for short text
        mock_controller.type.assert_called_once_with("Hi")
        mock_pyperclip.copy.assert_not_called()
    
    @patch('whisper_transcriber.text_inserter.pyperclip')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_insert_text_auto_method_long(self, mock_controller_class, mock_pyperclip, text_inserter):
        """Test auto method chooses clipboard for long text"""
        # Setup mock controller
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        long_text = "This is a very long text that exceeds the threshold for keyboard typing method"
        mock_pyperclip.paste.return_value = "original"
        
        text_inserter.insert_text(long_text, method=InsertMethod.AUTO)
        
        # Should use clipboard method for long text
        mock_pyperclip.copy.assert_any_call(long_text)
        mock_controller_class.assert_called()  # Controller was used for paste
    
    @patch('whisper_transcriber.text_inserter.pyperclip')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_insert_text_with_newlines(self, mock_controller_class, mock_pyperclip, text_inserter):
        """Test auto method chooses clipboard for text with newlines"""
        # Setup mock controller
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        text_with_newlines = "Line 1\nLine 2"
        mock_pyperclip.paste.return_value = "original"
        
        text_inserter.insert_text(text_with_newlines, method=InsertMethod.AUTO)
        
        # Should use clipboard method for text with newlines
        mock_pyperclip.copy.assert_any_call(text_with_newlines)
        mock_controller_class.assert_called()  # Controller was used for paste
    
    @patch('whisper_transcriber.text_inserter.pyperclip')
    def test_clipboard_restoration(self, mock_pyperclip, text_inserter):
        """Test clipboard is properly restored after insertion"""
        original_content = "original clipboard content"
        mock_pyperclip.paste.return_value = original_content
        
        with patch('whisper_transcriber.text_inserter.keyboard.Controller'):
            text_inserter._clipboard_method("new text")
        
        # Verify restoration
        calls = mock_pyperclip.copy.call_args_list
        assert calls[0][0][0] == "new text"  # First copy new text
        assert calls[1][0][0] == original_content  # Then restore original
    
    @patch('whisper_transcriber.text_inserter.pyperclip')
    def test_clipboard_method_empty_original(self, mock_pyperclip, text_inserter):
        """Test clipboard method handles empty original clipboard"""
        mock_pyperclip.paste.return_value = ""
        
        with patch('whisper_transcriber.text_inserter.keyboard.Controller'):
            text_inserter._clipboard_method("new text")
        
        # Should still work with empty clipboard
        mock_pyperclip.copy.assert_any_call("new text")
        mock_pyperclip.copy.assert_any_call("")
    
    @patch('whisper_transcriber.text_inserter.pyperclip')
    def test_clipboard_method_paste_error(self, mock_pyperclip, text_inserter):
        """Test clipboard method handles paste errors gracefully"""
        mock_pyperclip.paste.side_effect = Exception("Clipboard error")
        
        with patch('whisper_transcriber.text_inserter.keyboard.Controller'):
            # Should not raise exception
            text_inserter._clipboard_method("new text")
        
        # Should still copy new text (first call is to copy the new text)
        assert mock_pyperclip.copy.call_count == 2
        assert mock_pyperclip.copy.call_args_list[0][0][0] == "new text"
        # Second call is to restore the clipboard (empty string due to paste error)
        assert mock_pyperclip.copy.call_args_list[1][0][0] == ""
    
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_keyboard_method_special_characters(self, mock_controller_class, text_inserter):
        """Test keyboard method handles special characters"""
        # Setup mock controller
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        text_with_special = "Hello! @#$%^&*()"
        text_inserter._keyboard_method(text_with_special)
        
        mock_controller.type.assert_called_once_with(text_with_special)
    
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_keyboard_method_unicode(self, mock_controller_class, text_inserter):
        """Test keyboard method handles unicode characters"""
        # Setup mock controller
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        unicode_text = "Hello 世界 🌍"
        text_inserter._keyboard_method(unicode_text)
        
        mock_controller.type.assert_called_once_with(unicode_text)
    
    def test_invalid_insert_method(self, text_inserter):
        """Test handling of invalid insert method"""
        with pytest.raises(ValueError):
            text_inserter.insert_text("text", method="invalid_method")
    
    @patch('whisper_transcriber.text_inserter.time.sleep')
    @patch('whisper_transcriber.text_inserter.pyperclip')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_timing_delays(self, mock_controller_class, mock_pyperclip, mock_sleep, text_inserter):
        """Test proper timing delays are used"""
        # Setup mock controller with context manager support
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        mock_controller.pressed = Mock()
        mock_controller.pressed.return_value.__enter__ = Mock()
        mock_controller.pressed.return_value.__exit__ = Mock()
        
        mock_pyperclip.paste.return_value = "original"
        
        text_inserter._clipboard_method("text")
        
        # Verify delays are called
        assert mock_sleep.call_count >= 2  # At least 2 delays
    
    @patch('whisper_transcriber.text_inserter.platform.system')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_platform_specific_shortcuts_mac(self, mock_controller_class, mock_platform, text_inserter):
        """Test platform-specific keyboard shortcuts on macOS"""
        # Setup mock controller
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        mock_controller.pressed = Mock()
        mock_controller.pressed.return_value.__enter__ = Mock()
        mock_controller.pressed.return_value.__exit__ = Mock()
        
        mock_platform.return_value = "Darwin"
        
        with patch('whisper_transcriber.text_inserter.pyperclip'):
            text_inserter._clipboard_method("text")
        
        # Verify CMD+V was pressed
        mock_controller.pressed.assert_called()  # Should be called with Key.cmd
        mock_controller.press.assert_called_with('v')
        mock_controller.release.assert_called_with('v')
    
    @patch('whisper_transcriber.text_inserter.platform.system')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_platform_specific_shortcuts_windows(self, mock_controller_class, mock_platform, text_inserter):
        """Test platform-specific keyboard shortcuts on Windows"""
        # Setup mock controller
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        mock_controller.pressed = Mock()
        mock_controller.pressed.return_value.__enter__ = Mock()
        mock_controller.pressed.return_value.__exit__ = Mock()
        
        mock_platform.return_value = "Windows"
        
        with patch('whisper_transcriber.text_inserter.pyperclip'):
            text_inserter._clipboard_method("text")
        
        # Verify CTRL+V was pressed
        mock_controller.pressed.assert_called()  # Should be called with Key.ctrl
        mock_controller.press.assert_called_with('v')
        mock_controller.release.assert_called_with('v')
    
    @patch('whisper_transcriber.text_inserter.platform.system')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_platform_specific_shortcuts_linux(self, mock_controller_class, mock_platform, text_inserter):
        """Test platform-specific keyboard shortcuts on Linux"""
        # Setup mock controller
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        mock_controller.pressed = Mock()
        mock_controller.pressed.return_value.__enter__ = Mock()
        mock_controller.pressed.return_value.__exit__ = Mock()
        
        mock_platform.return_value = "Linux"
        
        with patch('whisper_transcriber.text_inserter.pyperclip'):
            text_inserter._clipboard_method("text")
        
        # Verify CTRL+V was pressed
        mock_controller.pressed.assert_called()  # Should be called with Key.ctrl
        mock_controller.press.assert_called_with('v')
        mock_controller.release.assert_called_with('v')
    
    def test_empty_text_insertion(self, text_inserter):
        """Test insertion of empty text"""
        with patch('whisper_transcriber.text_inserter.keyboard.Controller') as mock_controller_class:
            # Setup mock controller
            mock_controller = Mock()
            mock_controller_class.return_value = mock_controller
            
            text_inserter.insert_text("", method=InsertMethod.KEYBOARD)
            mock_controller.type.assert_called_once_with("")
    
    @patch('whisper_transcriber.text_inserter.pyperclip')
    @patch('whisper_transcriber.text_inserter.keyboard.Controller')
    def test_fallback_on_clipboard_error(self, mock_controller_class, mock_pyperclip, text_inserter):
        """Test fallback to keyboard method on clipboard error"""
        # Setup mock controller
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        mock_pyperclip.copy.side_effect = Exception("Clipboard error")
        
        # Should fallback to keyboard method
        text_inserter.insert_text("Hello", method=InsertMethod.CLIPBOARD)
        
        # Verify fallback to keyboard
        mock_controller.type.assert_called_once_with("Hello")
    
    @patch('whisper_transcriber.text_inserter.pyperclip')
    def test_get_clipboard_content(self, mock_pyperclip, text_inserter):
        """Test getting current clipboard content"""
        mock_pyperclip.paste.return_value = "clipboard content"
        
        content = text_inserter.get_clipboard_content()
        
        assert content == "clipboard content"
        mock_pyperclip.paste.assert_called_once()