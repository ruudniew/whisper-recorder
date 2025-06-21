import logging
import rumps
from typing import Optional

from .config import ConfigManager
from .audio_capture import AudioCapture
from .hotkey_manager import HotkeyManager
from .startup_manager import StartupManager


logger = logging.getLogger(__name__)


class SimplePreferencesWindow:
    """Simple preferences dialog using rumps.Window"""

    def __init__(
        self,
        config_manager: ConfigManager,
        audio_capture: AudioCapture,
        hotkey_manager: HotkeyManager,
    ):
        """Initialize preferences window

        Args:
            config_manager: Application configuration manager
            audio_capture: Audio capture instance
            hotkey_manager: Hotkey manager instance
        """
        self.config_manager = config_manager
        self.audio_capture = audio_capture
        self.hotkey_manager = hotkey_manager
        self.startup_manager = StartupManager()

    def show(self):
        """Show the preferences dialog"""
        try:
            # Get current settings
            current_hotkey = self.config_manager.get("hotkey", "cmd+shift+r")
            current_model = self.config_manager.get("model", "tiny.en")
            current_language = self.config_manager.get("language", "en")
            current_insertion = self.config_manager.get("insertion_method", "clipboard")

            # Check actual system state for start at login
            start_at_login = self.startup_manager.is_startup_enabled()
            # Update config if different from system state
            if start_at_login != self.config_manager.get("start_at_login", False):
                self.config_manager.set("start_at_login", start_at_login)
                self.config_manager.save()

            # Create a multi-field dialog using multiple windows
            # Hotkey setting
            window = rumps.Window(
                title="Recording Hotkey",
                message="Enter the hotkey combination (e.g., cmd+shift+r):",
                default_text=current_hotkey,
                ok="Next",
                cancel="Cancel",
                dimensions=(320, 100),
            )
            response = window.run()
            if response.clicked == 0:  # Cancel
                return
            new_hotkey = response.text

            # Model setting
            window = rumps.Window(
                title="Whisper Model",
                message="Choose model: tiny.en, base.en, small.en\n(larger = better quality, slower)",
                default_text=current_model,
                ok="Next",
                cancel="Cancel",
                dimensions=(320, 100),
            )
            response = window.run()
            if response.clicked == 0:  # Cancel
                return
            new_model = response.text

            # Language setting
            window = rumps.Window(
                title="Language",
                message="Enter language code (en, es, fr, de, etc):",
                default_text=current_language,
                ok="Next",
                cancel="Cancel",
                dimensions=(320, 100),
            )
            response = window.run()
            if response.clicked == 0:  # Cancel
                return
            new_language = response.text

            # Insertion method
            window = rumps.Window(
                title="Text Insertion Method",
                message="Choose method: clipboard, keyboard, or auto",
                default_text=current_insertion,
                ok="Next",
                cancel="Cancel",
                dimensions=(320, 100),
            )
            response = window.run()
            if response.clicked == 0:  # Cancel
                return
            new_insertion = response.text

            # Microphone selection
            devices = self.audio_capture.list_devices()
            device_names = ["default (System Default)"]
            device_map = {"default": None}

            for device in devices:
                # Include ALL input devices - virtual, webcams, headphones, etc.
                device_desc = (
                    f"{device.name} ({device.channels}ch, {device.sample_rate}Hz)"
                )
                if device.is_default:
                    device_desc += " [Current Default]"
                device_names.append(device_desc)
                device_map[device_desc] = device.id

            # Get current device setting
            current_device_id = self.config_manager.get("audio_device_id", None)
            current_selection = "default (System Default)"

            # Find current device in list
            if current_device_id is not None:
                for desc, dev_id in device_map.items():
                    if dev_id == current_device_id:
                        current_selection = desc
                        break

            # Show device list as text for user to choose
            device_list = "\n".join(
                [f"{i+1}. {name}" for i, name in enumerate(device_names)]
            )
            window = rumps.Window(
                title="Select Microphone",
                message=f"Available input devices:\n{device_list}\n\nEnter device name or number:",
                default_text=current_selection,
                ok="Save All",
                cancel="Cancel",
                dimensions=(400, 300),
            )
            response = window.run()
            if response.clicked == 0:  # Cancel
                return

            # Parse selection
            selected_text = response.text.strip()
            selected_device_id = None

            # Check if user entered a number
            try:
                device_index = int(selected_text) - 1
                if 0 <= device_index < len(device_names):
                    selected_device_id = device_map[device_names[device_index]]
            except ValueError:
                # User entered text, find matching device
                for desc, dev_id in device_map.items():
                    if selected_text.lower() in desc.lower():
                        selected_device_id = dev_id
                        break

            # Save all settings
            self.config_manager.set("hotkey", new_hotkey)
            self.config_manager.set("model", new_model)
            self.config_manager.set("language", new_language)
            self.config_manager.set("insertion_method", new_insertion)
            self.config_manager.set("audio_device_id", selected_device_id)

            # Update old audio_device setting for compatibility
            if selected_device_id is None:
                self.config_manager.set("audio_device", "default")
            else:
                self.config_manager.set("audio_device", str(selected_device_id))

            # Start at login preference
            response = rumps.alert(
                title="Start at Login",
                message="Launch Whisper Transcriber when you log in to your Mac?",
                ok="Yes",
                cancel="No",
            )
            start_at_login = response == 1
            self.config_manager.set("start_at_login", start_at_login)

            # Apply the startup setting
            if self.startup_manager.toggle_startup(start_at_login):
                logger.info(f"Successfully set start at login to: {start_at_login}")
            else:
                logger.error("Failed to update start at login setting")
                rumps.alert(
                    title="Warning",
                    message="Could not update login item. You may need to set this manually in System Preferences.",
                )

            # Save to disk
            self.config_manager.save()

            # Show confirmation
            rumps.notification(
                title="Preferences Saved",
                subtitle="",
                message="Your settings have been saved. Some changes may require a restart.",
            )

            logger.info("Preferences saved successfully")

        except Exception as e:
            logger.error(f"Failed to show preferences window: {e}")
            rumps.alert(title="Error", message=f"Failed to show preferences: {str(e)}")
