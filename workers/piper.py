"""
Piper worker (OHF-Voice piper-tts, ONNX). Default voice en_US-lessac-medium.

Piper needs a voice: a named voice (auto-downloaded on first use) or a local
.onnx model plus its matching .onnx.json config. `load(model=...)` accepts a
path to a local voice -- a .onnx file, or a directory containing exactly one
-- which is what a model_source override provides; anything else is treated
as a voice name to auto-download into a module-local cache. Piper synthesizes
one 16-bit mono float chunk per sentence; the worker normalizes those frames
through numpy and exports a real mp3 via pydub.

Note: voices are ~60MB, downloaded at load time (inside the service's
per-worker try/except), so a network failure just disables piper rather than
aborting synthesis. Target: piper-tts 1.7.x (PiperVoice.synthesize yields
AudioChunk with sample_rate / audio_float_array / audio_int16_bytes).
"""

import numpy as np
from pathlib import Path

from pydub import AudioSegment

# (pip package, importable name) pairs the service auto-installs when missing.
REQUIRED_PIP = [("piper-tts", "piper")]

DEFAULT_VOICE = "en_US-lessac-medium"
# Module-local cache for auto-downloaded voices (next to clips/, config.json).
VOICES_DIR = Path(__file__).resolve().parent.parent / "piper_voices"

# load() accepts a local voice (.onnx or a dir containing one) via model_source.
MODEL_SOURCE_SUPPORTED = True


def _resolve_voice(model_source):
    """Map a model_source to a local .onnx path, downloading named voices."""
    if model_source:
        source = Path(model_source)
        if source.is_dir():
            onnx_files = sorted(source.glob("*.onnx"))
            if not onnx_files:
                raise RuntimeError(f"No .onnx voice found in {source}")
            return onnx_files[0]
        if source.suffix == ".onnx" and source.exists():
            return source
        voice_name = model_source
    else:
        voice_name = DEFAULT_VOICE

    # A named voice: auto-download the .onnx + .onnx.json into the cache.
    from piper.download_voices import download_voice

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    download_voice(voice_name, VOICES_DIR)
    return VOICES_DIR / f"{voice_name}.onnx"


def load(model=None):
    from piper import PiperVoice

    onnx_path = _resolve_voice(model)
    # PiperVoice.load() derives the config path as "<model>.json" by default.
    return PiperVoice.load(onnx_path)


def synthesize(model, message: str, output_path: str) -> None:
    chunks = list(model.synthesize(message))
    if not chunks:
        raise RuntimeError("Piper produced no audio for the message.")

    sample_rate = chunks[0].sample_rate
    audio = np.concatenate([c.audio_float_array for c in chunks])
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    segment = AudioSegment(
        audio.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    segment.export(output_path, format="mp3")