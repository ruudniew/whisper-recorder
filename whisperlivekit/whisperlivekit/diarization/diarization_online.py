import asyncio
import re
import threading
import numpy as np
import logging
import time
from queue import SimpleQueue, Empty

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sources import AudioSource
from whisperlivekit.timed_objects import SpeakerSegment
from diart.sources import MicrophoneAudioSource
from rx.core import Observer
from typing import Tuple, Any, List
from pyannote.core import Annotation
import diart.models as m

logger = logging.getLogger(__name__)

def extract_number(s: str) -> int:
    m = re.search(r'\d+', s)
    return int(m.group()) if m else None

class DiarizationObserver(Observer):
    """Observer that logs all data emitted by the diarization pipeline and stores speaker segments."""
    
    def __init__(self):
        self.speaker_segments = []
        self.processed_time = 0
        self.segment_lock = threading.Lock()
    
    def on_next(self, value: Tuple[Annotation, Any]):
        annotation, audio = value
        
        logger.debug("\n--- New Diarization Result ---")
        
        duration = audio.extent.end - audio.extent.start
        logger.debug(f"Audio segment: {audio.extent.start:.2f}s - {audio.extent.end:.2f}s (duration: {duration:.2f}s)")
        logger.debug(f"Audio shape: {audio.data.shape}")
        
        with self.segment_lock:
            if audio.extent.end > self.processed_time:
                self.processed_time = audio.extent.end            
            if annotation and len(annotation._labels) > 0:
                logger.debug("\nSpeaker segments:")
                for speaker, label in annotation._labels.items():
                    for start, end in zip(label.segments_boundaries_[:-1], label.segments_boundaries_[1:]):
                        print(f"  {speaker}: {start:.2f}s-{end:.2f}s")
                        self.speaker_segments.append(SpeakerSegment(
                            speaker=speaker,
                            start=start,
                            end=end
                        ))
            else:
                logger.debug("\nNo speakers detected in this segment")
                
    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.speaker_segments.copy()
    
    def clear_old_segments(self, older_than: float = 30.0):
        """Clear segments older than the specified time."""
        with self.segment_lock:
            current_time = self.processed_time
            self.speaker_segments = [
                segment for segment in self.speaker_segments 
                if current_time - segment.end < older_than
            ]
    
    def on_error(self, error):
        """Handle an error in the stream."""
        logger.debug(f"Error in diarization stream: {error}")
        
    def on_completed(self):
        """Handle the completion of the stream."""
        logger.debug("Diarization stream completed")


class WebSocketAudioSource(AudioSource):
    """
    Buffers incoming audio and releases it in fixed-size chunks at regular intervals.
    """
    def __init__(self, uri: str = "websocket", sample_rate: int = 16000, block_duration: float = 0.5):
        super().__init__(uri, sample_rate)
        self.block_duration = block_duration
        self.block_size = int(np.rint(block_duration * sample_rate))
        self._queue = SimpleQueue()
        self._buffer = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._closed = False
        self._close_event = threading.Event()
        self._processing_thread = None
        self._last_chunk_time = time.time()

    def read(self):
        """Start processing buffered audio and emit fixed-size chunks."""
        self._processing_thread = threading.Thread(target=self._process_chunks)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        self._close_event.wait()
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)

    def _process_chunks(self):
        """Process audio from queue and emit fixed-size chunks at regular intervals."""
        while not self._closed:
            try:
                audio_chunk = self._queue.get(timeout=0.1)
                
                with self._buffer_lock:
                    self._buffer = np.concatenate([self._buffer, audio_chunk])
                    
                    while len(self._buffer) >= self.block_size:
                        chunk = self._buffer[:self.block_size]
                        self._buffer = self._buffer[self.block_size:]
                        
                        current_time = time.time()
                        time_since_last = current_time - self._last_chunk_time
                        if time_since_last < self.block_duration:
                            time.sleep(self.block_duration - time_since_last)
                        
                        chunk_reshaped = chunk.reshape(1, -1)
                        self.stream.on_next(chunk_reshaped)
                        self._last_chunk_time = time.time()
                        
            except Empty:
                with self._buffer_lock:
                    if len(self._buffer) > 0 and time.time() - self._last_chunk_time > self.block_duration:
                        padded_chunk = np.zeros(self.block_size, dtype=np.float32)
                        padded_chunk[:len(self._buffer)] = self._buffer
                        self._buffer = np.array([], dtype=np.float32)
                        
                        chunk_reshaped = padded_chunk.reshape(1, -1)
                        self.stream.on_next(chunk_reshaped)
                        self._last_chunk_time = time.time()
            except Exception as e:
                logger.error(f"Error in audio processing thread: {e}")
                self.stream.on_error(e)
                break
        
        with self._buffer_lock:
            if len(self._buffer) > 0:
                padded_chunk = np.zeros(self.block_size, dtype=np.float32)
                padded_chunk[:len(self._buffer)] = self._buffer
                chunk_reshaped = padded_chunk.reshape(1, -1)
                self.stream.on_next(chunk_reshaped)
        
        self.stream.on_completed()

    def close(self):
        if not self._closed:
            self._closed = True
            self._close_event.set()

    def push_audio(self, chunk: np.ndarray):
        """Add audio chunk to the processing queue."""
        if not self._closed:
            if chunk.ndim > 1:
                chunk = chunk.flatten()
            self._queue.put(chunk)
            logger.debug(f'Added chunk to queue with {len(chunk)} samples')


class DiartDiarization:
    def __init__(self, sample_rate: int = 16000, config : SpeakerDiarizationConfig = None, use_microphone: bool = False, block_duration: float = 0.5, segmentation_model_name: str = "pyannote/segmentation-3.0", embedding_model_name: str = "speechbrain/spkrec-ecapa-voxceleb"):
        segmentation_model = m.SegmentationModel.from_pretrained(segmentation_model_name)
        embedding_model = m.EmbeddingModel.from_pretrained(embedding_model_name)
        
        if config is None:
            config = SpeakerDiarizationConfig(
                segmentation=segmentation_model,
                embedding=embedding_model,
            )
        
        self.pipeline = SpeakerDiarization(config=config)        
        self.observer = DiarizationObserver()
        self.lag_diart = None
        
        if use_microphone:
            self.source = MicrophoneAudioSource(block_duration=block_duration)
            self.custom_source = None
        else:
            self.custom_source = WebSocketAudioSource(
                uri="websocket_source", 
                sample_rate=sample_rate,
                block_duration=block_duration
            )
            self.source = self.custom_source
            
        self.inference = StreamingInference(
            pipeline=self.pipeline,
            source=self.source,
            do_plot=False,
            show_progress=False,
        )
        self.inference.attach_observers(self.observer)
        asyncio.get_event_loop().run_in_executor(None, self.inference)

    async def diarize(self, pcm_array: np.ndarray):
        """
        Process audio data for diarization.
        Only used when working with WebSocketAudioSource.
        """
        if self.custom_source:
            self.custom_source.push_audio(pcm_array)            
        self.observer.clear_old_segments()        
        return self.observer.get_segments()

    def close(self):
        """Close the audio source."""
        if self.custom_source:
            self.custom_source.close()

    def assign_speakers_to_tokens(self, end_attributed_speaker, tokens: list, use_punctuation_split: bool = False) -> float:
        """
        Assign speakers to tokens based on timing overlap with speaker segments.
        Uses the segments collected by the observer.
        
        If use_punctuation_split is True, uses punctuation marks to refine speaker boundaries.
        """
        segments = self.observer.get_segments()
        
        # Debug logging
        logger.debug(f"assign_speakers_to_tokens called with {len(tokens)} tokens")
        logger.debug(f"Available segments: {len(segments)}")
        for i, seg in enumerate(segments[:5]):  # Show first 5 segments
            logger.debug(f"  Segment {i}: {seg.speaker} [{seg.start:.2f}-{seg.end:.2f}]")
        
        if not self.lag_diart and segments and tokens:
            self.lag_diart = segments[0].start - tokens[0].start
        for token in tokens:
            for segment in segments:
                if not (segment.end <= token.start + self.lag_diart or segment.start >= token.end + self.lag_diart):
                    token.speaker = extract_number(segment.speaker) + 1
                    end_attributed_speaker = max(token.end, end_attributed_speaker)
        
        if use_punctuation_split and len(tokens) > 1:
            punctuation_marks = {'.', '!', '?'}
            
            print("Here are the tokens:", 
                  [(t.text, t.start, t.end, t.speaker) for t in tokens[:10]])
            
            segment_map = []
            for segment in segments:
                speaker_num = extract_number(segment.speaker) + 1
                segment_map.append((segment.start, segment.end, speaker_num))
            segment_map.sort(key=lambda x: x[0])
            
            i = 0
            while i < len(tokens):
                current_token = tokens[i]
                
                is_sentence_end = False
                if current_token.text and current_token.text.strip():
                    text = current_token.text.strip()
                    if text[-1] in punctuation_marks:
                        is_sentence_end = True
                        logger.debug(f"Token {i} ends sentence: '{current_token.text}' at {current_token.end:.2f}s")
                
                if is_sentence_end and current_token.speaker != -1:
                    punctuation_time = current_token.end
                    current_speaker = current_token.speaker
                    
                    j = i + 1
                    next_sentence_tokens = []                    
                    while j < len(tokens):
                        next_token = tokens[j]
                        next_sentence_tokens.append(j)
                        
                        # Check if this token ends the next sentence
                        if next_token.text and next_token.text.strip():
                            if next_token.text.strip()[-1] in punctuation_marks:
                                break
                        j += 1
                    
                    if next_sentence_tokens:
                        speaker_times = {}
                        
                        for idx in next_sentence_tokens:
                            token = tokens[idx]
                            # Find which segments overlap with this token
                            for seg_start, seg_end, seg_speaker in segment_map:
                                if not (seg_end <= token.start or seg_start >= token.end):
                                    # Calculate overlap duration
                                    overlap_start = max(seg_start, token.start)
                                    overlap_end = min(seg_end, token.end)
                                    overlap_duration = overlap_end - overlap_start
                                    
                                    if seg_speaker not in speaker_times:
                                        speaker_times[seg_speaker] = 0
                                    speaker_times[seg_speaker] += overlap_duration
                        
                        if speaker_times:
                            dominant_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
                            
                            if dominant_speaker != current_speaker:
                                logger.debug(f"  Speaker change after punctuation: {current_speaker} → {dominant_speaker}")
                                
                                for idx in next_sentence_tokens:
                                    if tokens[idx].speaker != dominant_speaker:
                                        logger.debug(f"    Reassigning token {idx} ('{tokens[idx].text}') to Speaker {dominant_speaker}")
                                        tokens[idx].speaker = dominant_speaker
                                        end_attributed_speaker = max(tokens[idx].end, end_attributed_speaker)
                            else:
                                for idx in next_sentence_tokens:
                                    if tokens[idx].speaker == -1:
                                        tokens[idx].speaker = current_speaker
                                        end_attributed_speaker = max(tokens[idx].end, end_attributed_speaker)
                
                i += 1         
        
        return end_attributed_speaker
