import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import time

from whisper_transcriber.hotkey_manager import HotkeyManager, HotkeyError


class TestHotkeyManager:
    """Test suite for HotkeyManager class"""
    
    @pytest.fixture
    def hotkey_manager(self):
        """Create HotkeyManager instance"""
        return HotkeyManager()
    
    def test_init(self, hotkey_manager):
        """Test HotkeyManager initialization"""
        assert hotkey_manager.hotkeys == {}
        assert hotkey_manager.listener is None
        assert hotkey_manager._is_listening is False
    
    def test_register_hotkey_success(self, hotkey_manager):
        """Test successful hotkey registration"""
        callback = Mock()
        
        hotkey_manager.register_hotkey("cmd+shift+r", callback)
        
        assert "cmd+shift+r" in hotkey_manager.hotkeys
        assert hotkey_manager.hotkeys["cmd+shift+r"] == callback
    
    def test_register_hotkey_duplicate(self, hotkey_manager):
        """Test registering duplicate hotkey raises error"""
        callback1 = Mock()
        callback2 = Mock()
        
        hotkey_manager.register_hotkey("cmd+shift+r", callback1)
        
        with pytest.raises(HotkeyError, match="already registered"):
            hotkey_manager.register_hotkey("cmd+shift+r", callback2)
    
    def test_register_hotkey_invalid_format(self, hotkey_manager):
        """Test registering invalid hotkey format"""
        callback = Mock()
        
        # Empty hotkey
        with pytest.raises(ValueError, match="Invalid hotkey"):
            hotkey_manager.register_hotkey("", callback)
        
        # None hotkey
        with pytest.raises(ValueError, match="Invalid hotkey"):
            hotkey_manager.register_hotkey(None, callback)
    
    def test_unregister_hotkey_success(self, hotkey_manager):
        """Test successful hotkey unregistration"""
        callback = Mock()
        hotkey_manager.register_hotkey("cmd+shift+r", callback)
        
        hotkey_manager.unregister_hotkey("cmd+shift+r")
        
        assert "cmd+shift+r" not in hotkey_manager.hotkeys
    
    def test_unregister_hotkey_not_found(self, hotkey_manager):
        """Test unregistering non-existent hotkey"""
        # Should not raise error
        hotkey_manager.unregister_hotkey("cmd+shift+x")
    
    @patch('whisper_transcriber.hotkey_manager.keyboard.GlobalHotKeys')
    def test_start_listening_success(self, mock_global_hotkeys, hotkey_manager):
        """Test starting hotkey listener"""
        # Setup mocks
        mock_listener = MagicMock()
        mock_global_hotkeys.return_value = mock_listener
        
        # Register some hotkeys
        callback1 = Mock()
        callback2 = Mock()
        hotkey_manager.register_hotkey("cmd+shift+r", callback1)
        hotkey_manager.register_hotkey("cmd+shift+t", callback2)
        
        # Start listening
        hotkey_manager.start_listening()
        
        # Verify listener was created with correct hotkeys
        mock_global_hotkeys.assert_called_once()
        hotkeys_dict = mock_global_hotkeys.call_args[0][0]
        assert '<cmd>+<shift>+r' in hotkeys_dict
        assert '<cmd>+<shift>+t' in hotkeys_dict
        
        # Verify listener was started
        mock_listener.start.assert_called_once()
        assert hotkey_manager._is_listening is True
    
    @patch('whisper_transcriber.hotkey_manager.keyboard.GlobalHotKeys')
    def test_start_listening_already_listening(self, mock_global_hotkeys, hotkey_manager):
        """Test starting listener when already listening"""
        hotkey_manager._is_listening = True
        hotkey_manager.listener = Mock()
        
        # Should not create new listener
        hotkey_manager.start_listening()
        
        mock_global_hotkeys.assert_not_called()
    
    @patch('whisper_transcriber.hotkey_manager.keyboard.GlobalHotKeys')
    def test_stop_listening(self, mock_global_hotkeys, hotkey_manager):
        """Test stopping hotkey listener"""
        # Setup mock listener
        mock_listener = MagicMock()
        mock_global_hotkeys.return_value = mock_listener
        
        # Start listening first
        hotkey_manager.register_hotkey("cmd+shift+r", Mock())
        hotkey_manager.start_listening()
        
        # Stop listening
        hotkey_manager.stop_listening()
        
        # Verify listener was stopped
        mock_listener.stop.assert_called_once()
        assert hotkey_manager._is_listening is False
        assert hotkey_manager.listener is None
    
    def test_stop_listening_not_started(self, hotkey_manager):
        """Test stopping listener when not started"""
        # Should not raise error
        hotkey_manager.stop_listening()
        assert hotkey_manager._is_listening is False
    
    def test_parse_hotkey_combination(self, hotkey_manager):
        """Test parsing hotkey combinations"""
        # Test various formats
        assert hotkey_manager._parse_hotkey("cmd+shift+r") == "<cmd>+<shift>+r"
        assert hotkey_manager._parse_hotkey("ctrl+alt+delete") == "<ctrl>+<alt>+<delete>"
        assert hotkey_manager._parse_hotkey("cmd+a") == "<cmd>+a"
        assert hotkey_manager._parse_hotkey("f1") == "f1"
        assert hotkey_manager._parse_hotkey("shift+f1") == "<shift>+f1"
    
    def test_parse_hotkey_case_insensitive(self, hotkey_manager):
        """Test hotkey parsing is case insensitive for modifiers"""
        assert hotkey_manager._parse_hotkey("CMD+SHIFT+R") == "<cmd>+<shift>+r"
        assert hotkey_manager._parse_hotkey("Ctrl+Alt+Delete") == "<ctrl>+<alt>+<delete>"
    
    def test_parse_hotkey_platform_specific(self, hotkey_manager):
        """Test platform-specific key mappings"""
        # Command key variations
        assert hotkey_manager._parse_hotkey("command+a") == "<cmd>+a"
        assert hotkey_manager._parse_hotkey("win+a") == "<cmd>+a"  # Windows key mapped to cmd
        
        # Option key variations
        assert hotkey_manager._parse_hotkey("option+a") == "<alt>+a"
        assert hotkey_manager._parse_hotkey("opt+a") == "<alt>+a"
    
    @patch('whisper_transcriber.hotkey_manager.keyboard.GlobalHotKeys')
    def test_hotkey_callback_execution(self, mock_global_hotkeys, hotkey_manager):
        """Test hotkey callbacks are executed correctly"""
        # Setup callback
        callback = Mock()
        hotkey_manager.register_hotkey("cmd+shift+r", callback)
        
        # Start listening
        hotkey_manager.start_listening()
        
        # Get the wrapped callback
        hotkeys_dict = mock_global_hotkeys.call_args[0][0]
        wrapped_callback = hotkeys_dict['<cmd>+<shift>+r']
        
        # Execute wrapped callback
        wrapped_callback()
        
        # Verify original callback was called
        callback.assert_called_once()
    
    @patch('whisper_transcriber.hotkey_manager.keyboard.GlobalHotKeys')
    def test_hotkey_callback_error_handling(self, mock_global_hotkeys, hotkey_manager):
        """Test error handling in hotkey callbacks"""
        # Setup callback that raises error
        callback = Mock(side_effect=Exception("Callback error"))
        hotkey_manager.register_hotkey("cmd+shift+r", callback)
        
        # Start listening
        hotkey_manager.start_listening()
        
        # Get wrapped callback
        hotkeys_dict = mock_global_hotkeys.call_args[0][0]
        wrapped_callback = hotkeys_dict['<cmd>+<shift>+r']
        
        # Execute should not raise error
        wrapped_callback()
        
        # Callback was still called
        callback.assert_called_once()
    
    def test_get_registered_hotkeys(self, hotkey_manager):
        """Test getting list of registered hotkeys"""
        hotkey_manager.register_hotkey("cmd+shift+r", Mock())
        hotkey_manager.register_hotkey("cmd+shift+t", Mock())
        
        hotkeys = hotkey_manager.get_registered_hotkeys()
        
        assert len(hotkeys) == 2
        assert "cmd+shift+r" in hotkeys
        assert "cmd+shift+t" in hotkeys
    
    def test_is_hotkey_registered(self, hotkey_manager):
        """Test checking if hotkey is registered"""
        hotkey_manager.register_hotkey("cmd+shift+r", Mock())
        
        assert hotkey_manager.is_hotkey_registered("cmd+shift+r") is True
        assert hotkey_manager.is_hotkey_registered("cmd+shift+t") is False
    
    def test_clear_all_hotkeys(self, hotkey_manager):
        """Test clearing all registered hotkeys"""
        hotkey_manager.register_hotkey("cmd+shift+r", Mock())
        hotkey_manager.register_hotkey("cmd+shift+t", Mock())
        
        hotkey_manager.clear_all_hotkeys()
        
        assert len(hotkey_manager.hotkeys) == 0
    
    @patch('whisper_transcriber.hotkey_manager.keyboard.GlobalHotKeys')
    def test_restart_listener_after_hotkey_change(self, mock_global_hotkeys, hotkey_manager):
        """Test listener is restarted when hotkeys change while listening"""
        mock_listener = MagicMock()
        mock_global_hotkeys.return_value = mock_listener
        
        # Start listening
        hotkey_manager.register_hotkey("cmd+shift+r", Mock())
        hotkey_manager.start_listening()
        
        # Register new hotkey while listening
        hotkey_manager.register_hotkey("cmd+shift+t", Mock())
        
        # Should restart listener
        assert mock_listener.stop.call_count >= 1
        assert mock_global_hotkeys.call_count >= 2
    
    def test_thread_safety(self, hotkey_manager):
        """Test thread-safe operations"""
        callbacks = [Mock() for _ in range(10)]
        threads = []
        
        def register_hotkey(index):
            hotkey_manager.register_hotkey(f"cmd+shift+{index}", callbacks[index])
        
        # Register hotkeys from multiple threads
        for i in range(10):
            t = threading.Thread(target=register_hotkey, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All hotkeys should be registered
        assert len(hotkey_manager.hotkeys) == 10