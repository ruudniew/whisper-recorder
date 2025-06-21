import json
import logging
import subprocess
import threading
import time
import shutil
import os
import sys
from typing import Callable, Optional, Dict, Any

import websocket

from .models import ServerConfig


logger = logging.getLogger(__name__)

# Add local whisperlivekit to path if it exists
local_whisperlivekit = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "whisperlivekit"
)
if os.path.exists(local_whisperlivekit):
    sys.path.insert(0, local_whisperlivekit)
    logger.info(f"Using local whisperlivekit from {local_whisperlivekit}")


class TranscriptionError(Exception):
    """Exception raised for transcription-related errors"""

    pass


class TranscriptionService:
    """Manages WhisperLiveKit integration"""

    def __init__(self, server_config: ServerConfig):
        """Initialize TranscriptionService

        Args:
            server_config: Server configuration settings
        """
        self.server_config = server_config
        self.server_process: Optional[subprocess.Popen] = None
        self.websocket_client: Optional[websocket.WebSocketApp] = None
        self.is_connected = False
        self.transcription_callback: Optional[Callable[[str, bool], None]] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._audio_buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._last_send_time = time.time()

    def start_server(self) -> bool:
        """Launch WhisperLiveKit server subprocess

        Returns:
            True if server started successfully, False otherwise
        """
        with self._lock:
            # Check if already running
            if self.server_process and self.server_process.poll() is None:
                logger.info("Server already running")
                return True

            try:
                # Try to find whisperlivekit-server in venv or system
                whisperlivekit_cmd = None

                # First check if we're in a virtual environment
                if hasattr(sys, "real_prefix") or (
                    hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
                ):
                    # We're in a virtual environment
                    venv_bin = os.path.dirname(sys.executable)
                    venv_cmd = os.path.join(venv_bin, "whisperlivekit-server")
                    if os.path.exists(venv_cmd):
                        whisperlivekit_cmd = venv_cmd
                        logger.info(f"Found whisperlivekit-server in venv: {venv_cmd}")

                # If not found in venv, check system PATH
                if not whisperlivekit_cmd:
                    whisperlivekit_cmd = shutil.which("whisperlivekit-server")

                if not whisperlivekit_cmd:
                    logger.error("whisperlivekit-server command not found")
                    logger.error("Please install with: pip install whisperlivekit")
                    return False

                # Build command
                cmd = [
                    whisperlivekit_cmd,
                    "--host",
                    self.server_config.host,
                    "--port",
                    str(self.server_config.port),
                    "--model",
                    self.server_config.model,
                    "--lan",  # whisperlivekit uses --lan instead of --language
                    self.server_config.language,
                ]

                # whisperlivekit uses --no-vad to disable VAD (VAD is on by default)
                if not self.server_config.vad_enabled:
                    cmd.append("--no-vad")

                # Enable raw PCM mode to avoid WebM encoding/decoding
                cmd.append("--raw-pcm")

                logger.info(
                    f"Starting WhisperLiveKit server with command: {' '.join(cmd)}"
                )

                # Start server process
                self.server_process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                # Wait a bit for server to start
                time.sleep(1.5)

                # Check if process is still running
                if self.server_process.poll() is None:
                    logger.info(f"Server started with PID {self.server_process.pid}")
                    return True
                else:
                    # Get error output
                    stdout, stderr = self.server_process.communicate()
                    logger.error(f"Server process terminated immediately")
                    logger.error(f"Command was: {' '.join(cmd)}")
                    logger.error(f"stdout: {stdout}")
                    logger.error(f"stderr: {stderr}")
                    self.server_process = None
                    return False

            except Exception as e:
                logger.error(f"Failed to start server: {e}")
                self.server_process = None
                return False

    def connect_websocket(self) -> bool:
        """Establish WebSocket connection

        Returns:
            True if connection established, False otherwise
        """
        if not self.transcription_callback:
            logger.error("No transcription callback set")
            return False

        try:
            # Create WebSocket app
            self.websocket_client = websocket.WebSocketApp(
                self.server_config.websocket_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            # Run WebSocket in separate thread
            self._ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
            self._ws_thread.start()

            # Wait for connection with retries
            max_retries = 3
            retry_delay = 0.5

            for attempt in range(max_retries):
                if attempt > 0:
                    logger.info(
                        f"Retrying connection (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(retry_delay)

                    # Re-create WebSocket app for retry
                    self.websocket_client = websocket.WebSocketApp(
                        self.server_config.websocket_url,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close,
                    )

                    # Run WebSocket in separate thread
                    self._ws_thread = threading.Thread(
                        target=self._run_websocket, daemon=True
                    )
                    self._ws_thread.start()

                # Wait for connection
                timeout = 2
                start_time = time.time()
                while not self.is_connected and time.time() - start_time < timeout:
                    time.sleep(0.05)

                if self.is_connected:
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            return False

    def _run_websocket(self):
        """Run WebSocket client (in separate thread)"""
        try:
            self.websocket_client.run_forever()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False

    def send_audio_chunk(self, audio_data: bytes) -> None:
        """Stream audio to transcription server directly as raw PCM

        Args:
            audio_data: Raw audio data bytes (PCM format)
        """
        if not self.is_connected or not self.websocket_client:
            logger.debug(f"Cannot send audio: not connected")
            return

        try:
            # Send raw PCM data directly to server
            self.websocket_client.send(audio_data, opcode=websocket.ABNF.OPCODE_BINARY)
            logger.debug(f"Sent PCM chunk: {len(audio_data)} bytes")
        except Exception as e:
            logger.error(f"Failed to send PCM chunk: {e}")

    def handle_transcription(self, text: str, is_final: bool) -> None:
        """Process incoming transcription results

        Args:
            text: Transcribed text
            is_final: Whether this is a final transcription
        """
        if self.transcription_callback:
            try:
                self.transcription_callback(text, is_final)
            except Exception as e:
                logger.error(f"Error in transcription callback: {e}")

    def stop_server(self) -> None:
        """Stop the WhisperLiveKit server"""
        with self._lock:
            if not self.server_process:
                return

            try:
                logger.info("Stopping server...")
                self.server_process.terminate()

                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Server didn't stop gracefully, killing...")
                    self.server_process.kill()
                    self.server_process.wait()

                logger.info("Server stopped")
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
            finally:
                self.server_process = None

    def disconnect_websocket(self) -> None:
        """Close WebSocket connection"""
        if self.websocket_client:
            try:
                # Send empty buffer as stop signal (like the web client does)
                self.websocket_client.send(b"", opcode=websocket.ABNF.OPCODE_BINARY)
                logger.debug("Sent stop signal (empty buffer)")
                time.sleep(0.1)  # Give server time to process
                self.websocket_client.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket_client = None
                self.is_connected = False

        # Reset transcription tracking when disconnecting
        if hasattr(self, "_sent_texts"):
            self._sent_texts.clear()
        if hasattr(self, "_last_buffer_text"):
            self._last_buffer_text = ""
        if hasattr(self, "_last_buffer_content"):
            self._last_buffer_content = ""
        if hasattr(self, "_last_meaningful_transcription_time"):
            self._last_meaningful_transcription_time = None

    def _on_open(self, ws) -> None:
        """WebSocket open event handler"""
        logger.info("WebSocket connection opened")
        self.is_connected = True

    def _on_message(self, ws, message) -> None:
        """WebSocket message event handler"""
        try:
            # Parse JSON message
            data = json.loads(message)
            logger.info(f"Received WebSocket message: {data}")

            # WhisperLiveKit actual message format
            msg_type = data.get("type", "")
            status = data.get("status", "")

            # Handle different message types
            if msg_type == "ready_to_stop":
                logger.info("Server ready to stop - processing complete")
                return

            # Keep track of previously sent text to avoid duplicates
            if not hasattr(self, "_sent_texts"):
                self._sent_texts = set()
            if not hasattr(self, "_last_buffer_text"):
                self._last_buffer_text = ""
            if not hasattr(self, "_last_meaningful_transcription_time"):
                self._last_meaningful_transcription_time = None
            if not hasattr(self, "_silence_timeout_seconds"):
                self._silence_timeout_seconds = 3.0
            if not hasattr(self, "_last_buffer_content"):
                self._last_buffer_content = ""

            # Check for transcription in buffer_transcription
            buffer_text = data.get("buffer_transcription", "").strip()
            current_time = time.time()

            # Check for silence timeout - clear buffer if we've been silent too long
            if self._last_meaningful_transcription_time:
                time_since_last = (
                    current_time - self._last_meaningful_transcription_time
                )
                if time_since_last > self._silence_timeout_seconds:
                    logger.debug(
                        f"Silence detected for {time_since_last:.1f}s, clearing buffer tracking"
                    )
                    self._last_buffer_text = ""
                    self._last_buffer_content = ""
                    self._last_meaningful_transcription_time = None

            if buffer_text:
                logger.info(f"Buffer transcription: {buffer_text}")

                # Check if this is just a repetition of the same content
                if buffer_text == self._last_buffer_content:
                    logger.debug("Ignoring repeated buffer content")
                    return

                # Only send the new part of the buffer
                if buffer_text.startswith(self._last_buffer_text) and len(
                    buffer_text
                ) > len(self._last_buffer_text):
                    # Extract only the new text (don't strip to preserve spaces between words)
                    new_text = buffer_text[len(self._last_buffer_text) :]
                    if new_text.strip():  # Only check if non-empty after stripping
                        logger.info(f"New buffer text: {new_text}")
                        self.handle_transcription(new_text, False)
                        self._last_meaningful_transcription_time = current_time
                elif buffer_text != self._last_buffer_text:
                    # Complete buffer change, send it all
                    self.handle_transcription(buffer_text, False)
                    self._last_meaningful_transcription_time = current_time

                self._last_buffer_text = buffer_text
                self._last_buffer_content = buffer_text

            # Check for transcription in lines array
            lines = data.get("lines", [])

            for line in lines:
                if isinstance(line, dict):
                    # Line might have text field
                    line_text = line.get("text", "").strip()
                    if line_text and line_text not in self._sent_texts:
                        logger.info(f"Line transcription: {line_text}")
                        self.handle_transcription(line_text, True)
                        self._sent_texts.add(line_text)
                        self._last_meaningful_transcription_time = time.time()
                elif isinstance(line, str) and line.strip():
                    # Line might be just a string
                    line_text = line.strip()
                    if line_text not in self._sent_texts:
                        logger.info(f"Line transcription (str): {line_text}")
                        self.handle_transcription(line_text, True)
                        self._sent_texts.add(line_text)
                        self._last_meaningful_transcription_time = time.time()

            # Check if audio is being detected
            if status == "no_audio_detected":
                logger.debug("No audio detected by server")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def _on_error(self, ws, error) -> None:
        """WebSocket error event handler"""
        logger.error(f"WebSocket error: {error}")
        self.is_connected = False

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """WebSocket close event handler"""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.is_connected = False

    def is_server_running(self) -> bool:
        """Check if server process is running

        Returns:
            True if server is running, False otherwise
        """
        return self.server_process is not None and self.server_process.poll() is None

    def get_server_info(self) -> Dict[str, Any]:
        """Get server status and configuration

        Returns:
            Dictionary with server information
        """
        return {
            "host": self.server_config.host,
            "port": self.server_config.port,
            "model": self.server_config.model,
            "language": self.server_config.language,
            "websocket_url": self.server_config.websocket_url,
            "is_running": self.is_server_running(),
            "is_connected": self.is_connected,
        }

    def restart_server(self) -> bool:
        """Restart the server

        Returns:
            True if server restarted successfully
        """
        self.stop_server()
        time.sleep(0.5)  # Brief pause before restart
        return self.start_server()
