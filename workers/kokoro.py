"""
Kokoro worker (hexgrad/Kokoro-82M via the `kokoro` package). Default voice
af_heart, American English.

Note: the `kokoro` package is torch-based (KModel), not ONNX; the ONNX
variant lives in the separate `kokoro-onnx` package. `load(model=...)`
accepts an HF repo id override (e.g. hexgrad/Kokoro-82M-v1.1-zh) via
KPipeline(repo_id=...), so a model_source is supported. Voices are lazily
downloaded from the same repo id's voices/ dir. Output is 24 kHz mono float;
normalized through numpy and exported as a real mp3 via pydub.

Target: kokoro 0.9.x (KPipeline(lang_code=..., repo_id=...); a call yields
Result objects whose .audio is a CPU torch tensor).
"""

import numpy as np

from pydub import AudioSegment

# (pip package, importable name) pairs the service auto-installs when missing.
REQUIRED_PIP = [("kokoro", "kokoro"), ("misaki[en]", "misaki")]

MODEL_REPO = "hexgrad/Kokoro-82M"
LANG_CODE = "a"  # American English
DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000

# load() accepts an HF repo id via model_source (KPipeline(repo_id=...)).
MODEL_SOURCE_SUPPORTED = True


def load(model=None):
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=LANG_CODE, repo_id=model or MODEL_REPO)
    return pipeline, DEFAULT_VOICE


def synthesize(model, message: str, output_path: str) -> None:
    pipeline, voice = model

    chunks = []
    for result in pipeline(message, voice=voice):
        audio = result.audio
        if audio is None:
            continue
        # Real results are CPU torch tensors; the numpy fallback keeps this
        # independent of torch (a hard dep anyway) and testable with a stub.
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32))

    if not chunks:
        raise RuntimeError("Kokoro produced no audio for the message.")

    audio = np.concatenate(chunks)
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    segment = AudioSegment(
        audio.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1,
    )
    segment.export(output_path, format="mp3")