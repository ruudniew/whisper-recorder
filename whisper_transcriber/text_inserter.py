import logging
import platform
import time
from typing import Optional

import pyperclip
from pynput import keyboard

from .models import InsertMethod


logger = logging.getLogger(__name__)


class TextInserter:
    """Manages text insertion into active applications"""

    # Threshold for auto method selection
    AUTO_THRESHOLD_LENGTH = 50

    def __init__(self):
        """Initialize TextInserter"""
        self.original_clipboard = None
        self._platform = platform.system()

    def insert_text(
        self, text: str, method: InsertMethod = InsertMethod.CLIPBOARD
    ) -> None:
        """Insert text using specified method

        Args:
            text: Text to insert
            method: Insertion method to use
        """
        if isinstance(method, str):
            try:
                method = InsertMethod(method)
            except ValueError:
                raise ValueError(f"Invalid insert method: {method}")

        if method == InsertMethod.AUTO:
            # Choose method based on text characteristics
            if len(text) > self.AUTO_THRESHOLD_LENGTH or "\n" in text:
                method = InsertMethod.CLIPBOARD
            else:
                method = InsertMethod.KEYBOARD

        try:
            if method == InsertMethod.CLIPBOARD:
                self._clipboard_method(text)
            elif method == InsertMethod.KEYBOARD:
                self._keyboard_method(text)
        except Exception as e:
            logger.error(f"Failed to insert text with {method.value} method: {e}")
            # Try fallback method
            if method == InsertMethod.CLIPBOARD:
                logger.info("Falling back to keyboard method")
                self._keyboard_method(text)
            else:
                raise

    def _clipboard_method(self, text: str) -> None:
        """Insert via clipboard (most reliable)

        Args:
            text: Text to insert
        """
        try:
            logger.debug(f"Starting clipboard method for text: '{text}'")

            # Save original clipboard content
            try:
                self.original_clipboard = pyperclip.paste()
                logger.debug(f"Saved original clipboard content")
            except Exception:
                logger.warning("Could not retrieve original clipboard content")
                self.original_clipboard = ""

            # Copy new text to clipboard
            pyperclip.copy(text)
            logger.debug(f"Copied text to clipboard")
            time.sleep(0.1)  # Small delay for clipboard to update

            # Paste using platform-specific shortcut
            logger.debug(f"Executing paste keyboard shortcut")
            self._paste_with_keyboard()

            # Wait for paste to complete
            time.sleep(0.2)

            # Restore original clipboard
            self._restore_clipboard()
            logger.debug(f"Clipboard method completed successfully")

        except Exception as e:
            logger.error(f"Clipboard method failed: {e}")
            raise

    def _keyboard_method(self, text: str) -> None:
        """Type text directly (fallback)

        Args:
            text: Text to type
        """
        controller = keyboard.Controller()
        controller.type(text)

    def _restore_clipboard(self) -> None:
        """Restore original clipboard contents"""
        if self.original_clipboard is not None:
            try:
                pyperclip.copy(self.original_clipboard)
                self.original_clipboard = None
            except Exception as e:
                logger.warning(f"Could not restore clipboard: {e}")

    def _paste_with_keyboard(self) -> None:
        """Execute platform-specific paste keyboard shortcut"""
        controller = keyboard.Controller()

        if self._platform == "Darwin":  # macOS
            with controller.pressed(keyboard.Key.cmd):
                controller.press("v")
                controller.release("v")
        else:  # Windows, Linux
            with controller.pressed(keyboard.Key.ctrl):
                controller.press("v")
                controller.release("v")

    def get_clipboard_content(self) -> str:
        """Get current clipboard content

        Returns:
            Clipboard content as string
        """
        try:
            return pyperclip.paste()
        except Exception as e:
            logger.error(f"Failed to get clipboard content: {e}")
            return ""
