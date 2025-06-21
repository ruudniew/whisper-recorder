import logging
import os
import subprocess
import plistlib
from pathlib import Path


logger = logging.getLogger(__name__)


class StartupManager:
    """Manages macOS Login Items for auto-start at system startup"""

    def __init__(self, app_name: str = "WhisperTranscriber"):
        """Initialize startup manager

        Args:
            app_name: Name of the application
        """
        self.app_name = app_name
        self.bundle_path = self._get_app_bundle_path()

    def _get_app_bundle_path(self) -> str:
        """Get the path to the application bundle or script

        Returns:
            Path to the application
        """
        # Try to find the .app bundle first
        # This would be the case if the app is packaged
        app_paths = [
            f"/Applications/{self.app_name}.app",
            f"~/Applications/{self.app_name}.app",
            f"/Applications/Whisper Transcriber.app",
            f"~/Applications/Whisper Transcriber.app",
        ]

        for path in app_paths:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                return expanded

        # If no .app bundle, return the script path
        # This is for development or when run directly
        import sys

        return sys.executable

    def is_startup_enabled(self) -> bool:
        """Check if the app is set to start at login

        Returns:
            True if startup is enabled
        """
        try:
            # Use osascript to check login items
            script = """
            tell application "System Events"
                get the name of every login item
            end tell
            """

            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True
            )

            if result.returncode == 0:
                login_items = result.stdout.strip()
                return (
                    self.app_name in login_items or "WhisperTranscriber" in login_items
                )

        except Exception as e:
            logger.error(f"Failed to check startup status: {e}")

        return False

    def enable_startup(self) -> bool:
        """Add the app to macOS login items

        Returns:
            True if successful
        """
        try:
            # Use osascript to add to login items
            script = f"""
            tell application "System Events"
                make new login item at end with properties {{name:"{self.app_name}", path:"{self.bundle_path}", hidden:false}}
            end tell
            """

            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True
            )

            if result.returncode == 0:
                logger.info(f"Added {self.app_name} to login items")
                return True
            else:
                logger.error(f"Failed to add to login items: {result.stderr}")

        except Exception as e:
            logger.error(f"Failed to enable startup: {e}")

        return False

    def disable_startup(self) -> bool:
        """Remove the app from macOS login items

        Returns:
            True if successful
        """
        try:
            # Use osascript to remove from login items
            script = f"""
            tell application "System Events"
                delete login item "{self.app_name}"
            end tell
            """

            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True
            )

            # Also try with alternate name
            if result.returncode != 0:
                script = """
                tell application "System Events"
                    delete login item "WhisperTranscriber"
                end tell
                """

                result = subprocess.run(
                    ["osascript", "-e", script], capture_output=True, text=True
                )

            if result.returncode == 0:
                logger.info(f"Removed {self.app_name} from login items")
                return True
            else:
                # It might not exist, which is fine
                logger.debug(f"Could not remove from login items: {result.stderr}")
                return True

        except Exception as e:
            logger.error(f"Failed to disable startup: {e}")

        return False

    def toggle_startup(self, enabled: bool) -> bool:
        """Enable or disable startup at login

        Args:
            enabled: Whether to enable startup

        Returns:
            True if successful
        """
        if enabled:
            # First disable to avoid duplicates
            self.disable_startup()
            return self.enable_startup()
        else:
            return self.disable_startup()
