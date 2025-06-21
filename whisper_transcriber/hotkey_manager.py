import logging
import threading
from typing import Callable, Dict, List, Optional

from pynput import keyboard


logger = logging.getLogger(__name__)


class HotkeyError(Exception):
    """Exception raised for hotkey-related errors"""

    pass


class HotkeyManager:
    """Manages global keyboard shortcuts"""

    def __init__(self):
        """Initialize HotkeyManager"""
        self.hotkeys: Dict[str, Callable] = {}
        self.listener: Optional[keyboard.GlobalHotKeys] = None
        self._is_listening = False
        self._lock = threading.Lock()

    def register_hotkey(self, combination: str, callback: Callable) -> None:
        """Register a global hotkey

        Args:
            combination: Hotkey combination (e.g., "cmd+shift+r")
            callback: Function to call when hotkey is pressed

        Raises:
            ValueError: If hotkey format is invalid
            HotkeyError: If hotkey is already registered
        """
        if not combination:
            raise ValueError("Invalid hotkey: empty combination")

        with self._lock:
            if combination in self.hotkeys:
                raise HotkeyError(f"Hotkey '{combination}' is already registered")

            self.hotkeys[combination] = callback
            logger.info(f"Registered hotkey: {combination}")

            # Restart listener if already running
            if self._is_listening:
                self._restart_listener()

    def unregister_hotkey(self, combination: str) -> None:
        """Remove a hotkey registration

        Args:
            combination: Hotkey combination to unregister
        """
        with self._lock:
            if combination in self.hotkeys:
                del self.hotkeys[combination]
                logger.info(f"Unregistered hotkey: {combination}")

                # Restart listener if running
                if self._is_listening:
                    self._restart_listener()

    def start_listening(self) -> None:
        """Begin monitoring for hotkey events"""
        with self._lock:
            if self._is_listening:
                logger.warning("Hotkey listener already running")
                return

            if not self.hotkeys:
                logger.warning("No hotkeys registered")
                return

            # Convert hotkeys to pynput format
            pynput_hotkeys = {}
            for combination, callback in self.hotkeys.items():
                parsed_key = self._parse_hotkey(combination)
                # Wrap callback to handle errors
                pynput_hotkeys[parsed_key] = self._wrap_callback(callback)

            # Create and start listener
            self.listener = keyboard.GlobalHotKeys(pynput_hotkeys)
            self.listener.start()
            self._is_listening = True
            logger.info("Started hotkey listener")

    def stop_listening(self) -> None:
        """Stop monitoring for hotkey events"""
        with self._lock:
            if not self._is_listening:
                return

            if self.listener:
                self.listener.stop()
                self.listener = None

            self._is_listening = False
            logger.info("Stopped hotkey listener")

    def _restart_listener(self) -> None:
        """Restart the listener (called when hotkeys change)"""
        # Stop the old listener first (without lock to avoid deadlock)
        old_listener = self.listener
        self.listener = None
        self._is_listening = False

        # Release lock before stopping to avoid deadlock
        if old_listener:
            try:
                old_listener.stop()
            except Exception as e:
                logger.error(f"Error stopping listener: {e}")

        # Now start new listener
        # Create new hotkeys dict
        pynput_hotkeys = {}
        for combination, callback in self.hotkeys.items():
            parsed_key = self._parse_hotkey(combination)
            pynput_hotkeys[parsed_key] = self._wrap_callback(callback)

        # Create and start new listener
        self.listener = keyboard.GlobalHotKeys(pynput_hotkeys)
        self.listener.start()
        self._is_listening = True

    def _parse_hotkey(self, combination: str) -> str:
        """Parse hotkey combination to pynput format

        Args:
            combination: Human-readable hotkey (e.g., "cmd+shift+r")

        Returns:
            Pynput-formatted hotkey (e.g., "<cmd>+<shift>+r")
        """
        # Split combination
        parts = combination.lower().split("+")
        parsed_parts = []

        # Define modifier mappings
        modifiers = {
            "cmd": "<cmd>",
            "command": "<cmd>",
            "win": "<cmd>",  # Windows key
            "ctrl": "<ctrl>",
            "control": "<ctrl>",
            "alt": "<alt>",
            "option": "<alt>",  # macOS option
            "opt": "<alt>",
            "shift": "<shift>",
        }

        # Define special key mappings
        special_keys = {
            "space": "<space>",
            "spacebar": "<space>",
            "enter": "<enter>",
            "return": "<enter>",
            "tab": "<tab>",
            "esc": "<esc>",
            "escape": "<esc>",
            "backspace": "<backspace>",
            "delete": "<delete>",
            "up": "<up>",
            "down": "<down>",
            "left": "<left>",
            "right": "<right>",
        }

        for part in parts:
            part = part.strip()
            if part in modifiers:
                parsed_parts.append(modifiers[part])
            elif part in special_keys:
                parsed_parts.append(special_keys[part])
            else:
                # Regular key
                parsed_parts.append(part)

        return "+".join(parsed_parts)

    def _wrap_callback(self, callback: Callable) -> Callable:
        """Wrap callback to handle errors gracefully

        Args:
            callback: Original callback function

        Returns:
            Wrapped callback that handles errors
        """

        def wrapped():
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in hotkey callback: {e}")

        return wrapped

    def get_registered_hotkeys(self) -> List[str]:
        """Get list of registered hotkey combinations

        Returns:
            List of hotkey combination strings
        """
        with self._lock:
            return list(self.hotkeys.keys())

    def is_hotkey_registered(self, combination: str) -> bool:
        """Check if a hotkey is registered

        Args:
            combination: Hotkey combination to check

        Returns:
            True if hotkey is registered, False otherwise
        """
        with self._lock:
            return combination in self.hotkeys

    def clear_all_hotkeys(self) -> None:
        """Remove all registered hotkeys"""
        with self._lock:
            self.hotkeys.clear()
            logger.info("Cleared all hotkeys")

            # Stop listener if running
            if self._is_listening:
                self.stop_listening()
