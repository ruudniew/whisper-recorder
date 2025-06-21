"""
Simple WebM encoder using ffmpeg subprocess

NOTE: This module is no longer used as of the performance optimization update.
We now send raw PCM directly to the WhisperLiveKit server to avoid the
unnecessary encode/decode cycle. This file is kept for backwards compatibility.
"""

import logging
import subprocess
import threading
import queue
import shutil
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SimpleWebMEncoder:
    """
    Simple WebM encoder that uses ffmpeg to convert PCM to WebM
    Similar to how WhisperLiveKit uses ffmpeg to decode WebM to PCM

    DEPRECATED: No longer used, we send raw PCM directly to the server.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """Initialize the WebM encoder

        Args:
            sample_rate: Audio sample rate (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.on_chunk: Optional[Callable[[bytes], None]] = None
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._write_queue = queue.Queue()
        self._running = False

        # Check if ffmpeg is available
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    def start(self):
        """Start the ffmpeg process"""
        if self._running:
            return

        # Start ffmpeg process
        cmd = [
            "ffmpeg",
            "-f",
            "s16le",  # Input format: signed 16-bit little-endian
            "-ar",
            str(self.sample_rate),  # Sample rate
            "-ac",
            str(self.channels),  # Number of channels
            "-i",
            "pipe:0",  # Input from stdin
            "-c:a",
            "libopus",  # Use Opus codec
            "-b:a",
            "32k",  # Bitrate
            "-f",
            "webm",  # Output format
            "-live",
            "1",  # Live streaming mode
            "pipe:1",  # Output to stdout
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._running = True

            # Start reader thread
            self._reader_thread = threading.Thread(
                target=self._read_output, daemon=True
            )
            self._reader_thread.start()

            # Start writer thread
            self._writer_thread = threading.Thread(
                target=self._write_input, daemon=True
            )
            self._writer_thread.start()

            logger.info("WebM encoder started")

        except Exception as e:
            logger.error(f"Failed to start ffmpeg: {e}")
            raise

    def _read_output(self):
        """Read WebM chunks from ffmpeg stdout"""
        try:
            while self._running and self._process:
                # Read chunks from ffmpeg
                chunk = self._process.stdout.read(4096)
                if not chunk:
                    break

                # Call the callback if set
                if self.on_chunk:
                    try:
                        self.on_chunk(chunk)
                    except Exception as e:
                        logger.error(f"Error in on_chunk callback: {e}")

        except Exception as e:
            logger.error(f"Error reading from ffmpeg: {e}")
        finally:
            logger.debug("Reader thread exiting")

    def _write_input(self):
        """Write PCM data to ffmpeg stdin"""
        try:
            while self._running:
                try:
                    # Get data from queue with timeout
                    data = self._write_queue.get(timeout=0.1)
                    if data is None:  # Sentinel value to stop
                        break

                    if self._process and self._process.stdin:
                        self._process.stdin.write(data)
                        self._process.stdin.flush()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error writing to ffmpeg: {e}")
                    break

        except Exception as e:
            logger.error(f"Error in writer thread: {e}")
        finally:
            logger.debug("Writer thread exiting")

    def write_pcm(self, pcm_data: bytes):
        """Write PCM audio data to the encoder

        Args:
            pcm_data: Raw PCM audio data (s16le format)
        """
        if not self._running:
            self.start()

        self._write_queue.put(pcm_data)

    def stop(self):
        """Stop the encoder"""
        if not self._running:
            return

        self._running = False

        # Signal writer thread to stop
        self._write_queue.put(None)

        # Close stdin to signal ffmpeg to stop
        if self._process and self._process.stdin:
            try:
                self._process.stdin.close()
            except:
                pass

        # Wait for threads to finish
        if self._writer_thread:
            self._writer_thread.join(timeout=1)
        if self._reader_thread:
            self._reader_thread.join(timeout=1)

        # Terminate process if still running
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            except:
                pass

        self._process = None
        logger.info("WebM encoder stopped")
