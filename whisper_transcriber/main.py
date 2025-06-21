import logging
import os
import sys
from typing import Optional

import rumps

from .config import ConfigManager
from .audio_capture import AudioCapture
from .transcriber import TranscriptionService
from .text_inserter import TextInserter
from .hotkey_manager import HotkeyManager
from .models import ServerConfig, InsertMethod
from .startup_manager import StartupManager

# Import simple preferences window
from .preferences_simple import SimplePreferencesWindow


# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Set specific loggers to appropriate levels
logging.getLogger("whisper_transcriber.transcriber").setLevel(logging.DEBUG)
logging.getLogger("whisper_transcriber.text_inserter").setLevel(logging.DEBUG)
logging.getLogger("whisper_transcriber.audio_capture").setLevel(logging.DEBUG)


# Notifications removed - not working on modern macOS


class WhisperTranscriberApp(rumps.App):
    """Main application class managing menu bar presence"""

    def __init__(self):
        """Initialize the menu bar application"""
        # Get the path to resources
        self._resources_path = os.path.join(os.path.dirname(__file__), "resources")
        self._icon_ready = os.path.join(self._resources_path, "microphone.png")
        self._icon_recording = os.path.join(
            self._resources_path, "microphone-recording.png"
        )

        super().__init__("", icon=self._icon_ready, quit_button=None)

        # Initialize components
        self.config_manager = ConfigManager()

        # Get audio device from config
        audio_device_id = self.config_manager.get("audio_device_id", None)
        self.audio_capture = AudioCapture(device_id=audio_device_id)

        self.startup_manager = StartupManager()

        # Sync startup state with system
        current_startup_state = self.startup_manager.is_startup_enabled()
        saved_startup_state = self.config_manager.get("start_at_login", False)
        if current_startup_state != saved_startup_state:
            self.config_manager.set("start_at_login", current_startup_state)
            self.config_manager.save()

        # Create server config from settings
        server_config = ServerConfig(
            host="localhost",
            port=9090,
            model=self.config_manager.get("model", "tiny.en"),
            language=self.config_manager.get("language", "en"),
            vad_enabled=True,
            use_gpu=False,
        )
        self.transcription_service = TranscriptionService(server_config)
        self.transcription_service.transcription_callback = self._handle_transcription

        self.text_inserter = TextInserter()
        self.hotkey_manager = HotkeyManager()

        # State
        self.is_recording = False

        # Setup menu
        self._setup_menu()

        # Register hotkey
        hotkey = self.config_manager.get("hotkey", "cmd+shift+r")
        self.hotkey_manager.register_hotkey(hotkey, self.toggle_recording_hotkey)
        self.hotkey_manager.start_listening()

        # Start the transcription server immediately for faster recording
        logger.info("Starting WhisperLiveKit server in background...")
        if not self.transcription_service.start_server():
            logger.warning(
                "Failed to start server on launch - will retry on first recording"
            )
        else:
            logger.info("WhisperLiveKit server ready")

        logger.info("WhisperTranscriber initialized")

    def _setup_menu(self):
        """Set up the menu items"""
        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.toggle_recording),
            rumps.separator,
            rumps.MenuItem("Preferences...", callback=self.show_preferences),
            rumps.MenuItem("About", callback=self.show_about),
            rumps.separator,
            rumps.MenuItem("Quit", callback=self.quit_application),
        ]

    def toggle_recording(self, sender):
        """Start/stop recording with icon change"""
        if not self.is_recording:
            # Start recording
            logger.info("Starting recording...")

            # Start transcription server if not already running
            if not self.transcription_service.is_server_running():
                logger.info("Server not running, starting now...")
                if not self.transcription_service.start_server():
                    logger.error(
                        "Failed to start transcription server. Check if whisperlive-server is installed."
                    )
                    rumps.alert(
                        title="Server Error",
                        message="Could not start WhisperLiveKit server.\n\n"
                        "Please ensure whisperlive-server is installed:\n"
                        "pip install whisperlive\n\n"
                        "Check the logs for more details.",
                    )
                    return

            # Connect WebSocket
            if not self.transcription_service.connect_websocket():
                logger.error("Failed to connect to transcription server")
                self.transcription_service.stop_server()
                return

            # Start audio capture
            if not self.audio_capture.start_recording(self._handle_audio_chunk):
                logger.error("Failed to start audio capture")
                self.transcription_service.disconnect_websocket()
                self.transcription_service.stop_server()
                return

            # Update state
            self.is_recording = True
            self.icon = self._icon_recording  # Use recording icon
            self._update_menu_item()
            logger.info("Recording started - press hotkey to stop")

        else:
            # Stop recording
            logger.info("Stopping recording...")

            # Stop components
            self.audio_capture.stop_recording()
            self.transcription_service.disconnect_websocket()
            self.transcription_service.stop_server()

            # Reset transcription tracking
            if hasattr(self, "_inserted_text"):
                self._inserted_text = ""

            # Update state
            self.is_recording = False
            self.icon = self._icon_ready  # Back to ready icon
            self._update_menu_item()
            logger.info("Recording stopped - press hotkey to start")

    def toggle_recording_hotkey(self):
        """Hotkey handler for toggle recording"""
        self.toggle_recording(None)

    def _update_menu_item(self):
        """Update the recording menu item text"""
        # Find the Start Recording menu item
        for item in self.menu:
            if hasattr(item, "title"):
                # Handle both callable and non-callable title attributes
                try:
                    if callable(item.title):
                        title_str = str(item.title())
                    else:
                        title_str = str(item.title)

                    if "Recording" in title_str:
                        if self.is_recording:
                            item.title = "Stop Recording"
                        else:
                            item.title = "Start Recording"
                        break
                except:
                    pass

    def _handle_audio_chunk(self, audio_data: bytes):
        """Handle audio data from capture"""
        self.transcription_service.send_audio_chunk(audio_data)

    def _handle_transcription(self, text: str, is_final: bool):
        """Handle transcription results"""
        logger.debug(
            f"_handle_transcription called: text='{text}', is_final={is_final}"
        )

        # Keep track of what we've already inserted to prevent duplicates
        if not hasattr(self, "_inserted_text"):
            self._inserted_text = ""

        if text.strip():
            # For final transcriptions, check if this is cumulative
            if is_final:
                # Check if this text starts with what we've already inserted
                if text.startswith(self._inserted_text) and len(text) > len(
                    self._inserted_text
                ):
                    # Only insert the new part
                    new_text = text[len(self._inserted_text) :]
                    if new_text.strip():
                        method = self.config_manager.get(
                            "insertion_method", "clipboard"
                        )
                        logger.info(
                            f"Inserting new text: '{new_text}' using method: {method}"
                        )
                        try:
                            self.text_inserter.insert_text(
                                new_text, InsertMethod(method)
                            )
                            self._inserted_text = text  # Update what we've inserted
                            logger.info(f"Successfully inserted new text: {new_text}")
                        except Exception as e:
                            logger.error(f"Failed to insert text: {e}", exc_info=True)
                elif text != self._inserted_text:
                    # Completely new text, insert it all
                    method = self.config_manager.get("insertion_method", "clipboard")
                    logger.info(f"Inserting text: '{text}' using method: {method}")
                    try:
                        self.text_inserter.insert_text(text, InsertMethod(method))
                        self._inserted_text = text
                        logger.info(f"Successfully inserted text: {text}")
                    except Exception as e:
                        logger.error(f"Failed to insert text: {e}", exc_info=True)

    def show_preferences(self, sender):
        """Display preferences window"""
        logger.info("Opening preferences window...")
        prefs = SimplePreferencesWindow(
            self.config_manager, self.audio_capture, self.hotkey_manager
        )
        prefs.show()

    def show_about(self, sender):
        """Show about dialog"""
        rumps.alert(
            title="About Whisper Transcriber",
            message="Real-time transcription for macOS\n\n"
            "Version 0.1.0\n"
            "Using WhisperLiveKit for transcription",
        )

    def quit_application(self, sender):
        """Clean shutdown of all components"""
        logger.info("Shutting down...")

        # Stop recording if active
        if self.is_recording:
            self.audio_capture.stop_recording()
            self.transcription_service.disconnect_websocket()

        # Stop all services
        self.transcription_service.stop_server()
        self.hotkey_manager.stop_listening()

        # Quit app
        rumps.quit_application()

    def run(self):
        """Run the application"""
        # Run the app
        super().run()


def main():
    """Main entry point"""
    app = WhisperTranscriberApp()
    app.run()


if __name__ == "__main__":
    main()
