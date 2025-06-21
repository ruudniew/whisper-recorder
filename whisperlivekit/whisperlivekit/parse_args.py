
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Whisper FastAPI Online Server")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="The host address to bind the server to.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="The port number to bind the server to."
    )
    parser.add_argument(
        "--warmup-file",
        type=str,
        default=None,
        dest="warmup_file",
        help="""
        The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.
        If not set, uses https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav.
        If False, no warmup is performed.
        """,
    )

    parser.add_argument(
        "--confidence-validation",
        action="store_true",
        help="Accelerates validation of tokens using confidence scores. Transcription will be faster but punctuation might be less accurate.",
    )

    parser.add_argument(
        "--diarization",
        action="store_true",
        default=False,
        help="Enable speaker diarization.",
    )

    parser.add_argument(
        "--punctuation-split",
        action="store_true",
        default=False,
        help="Use punctuation marks from transcription to improve speaker boundary detection. Requires both transcription and diarization to be enabled.",
    )

    parser.add_argument(
        "--segmentation-model",
        type=str,
        default="pyannote/segmentation-3.0",
        help="Hugging Face model ID for pyannote.audio segmentation model.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="pyannote/embedding",
        help="Hugging Face model ID for pyannote.audio embedding model.",
    )

    parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="Disable transcription to only see live diarization results.",
    )
    
    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=0.5,
        help="Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        help="Name size of the Whisper model to use (default: tiny). Suggested values: tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo. The model is automatically downloaded from the model hub if not present in model cache dir.",
    )
    
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Overriding the default model cache dir where models downloaded from the hub are saved",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.",
    )
    parser.add_argument(
        "--lan",
        "--language",
        type=str,
        default="auto",
        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Transcribe or translate.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="faster-whisper",
        choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"],
        help="Load only this backend for Whisper processing.",
    )
    parser.add_argument(
        "--vac",
        action="store_true",
        default=False,
        help="Use VAC = voice activity controller. Recommended. Requires torch.",
    )
    parser.add_argument(
        "--vac-chunk-size", type=float, default=0.04, help="VAC sample size in seconds."
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD (voice activity detection).",
    )
    
    parser.add_argument(
        "--buffer_trimming",
        type=str,
        default="segment",
        choices=["sentence", "segment"],
        help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.',
    )
    parser.add_argument(
        "--buffer_trimming_sec",
        type=float,
        default=15,
        help="Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level",
        default="DEBUG",
    )
    parser.add_argument("--ssl-certfile", type=str, help="Path to the SSL certificate file.", default=None)
    parser.add_argument("--ssl-keyfile", type=str, help="Path to the SSL private key file.", default=None)
    
    parser.add_argument(
        "--raw-pcm",
        action="store_true",
        default=False,
        help="Accept raw PCM audio instead of WebM format (avoids FFmpeg decoding).",
    )


    args = parser.parse_args()
    
    args.transcription = not args.no_transcription
    args.vad = not args.no_vad    
    delattr(args, 'no_transcription')
    delattr(args, 'no_vad')
    
    return args
