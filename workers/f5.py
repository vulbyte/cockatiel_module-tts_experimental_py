"""
F5-TTS worker (SWivid/f5-tts, F5TTS_v1_Base).

F5-TTS is zero-shot and needs a reference clip plus its transcript. The
service only passes text + output_path, so this worker uses the reference wav
bundled inside the f5-tts package (infer/examples/basic/basic_ref_en.wav)
with its known transcript. `load(model=...)` accepts a checkpoint path (or an
hf:// URL, which cached_path resolves) via F5TTS(ckpt_file=...), so a
model_source override is supported. infer() returns the float32 wav + sample
rate; the worker normalizes and exports a real mp3 via pydub.

Target: f5-tts current main (F5TTS.infer signature is
infer(ref_file, ref_text, gen_text, ...) -- note the first argument is named
`ref_file`, not `ref_audio`).
"""

from importlib.resources import files

import numpy as np

from pydub import AudioSegment

# (pip package, importable name) pairs the service auto-installs when missing.
REQUIRED_PIP = [("f5-tts", "f5_tts")]

DEFAULT_MODEL = "F5TTS_v1_Base"
REF_TEXT = "Some call me nature, others call me mother nature."
# Bundled API demo reference clip; must match REF_TEXT above.
_REF_FILENAME = "infer/examples/basic/basic_ref_en.wav"

# load() accepts a checkpoint path / hf:// URL via ckpt_file=.
MODEL_SOURCE_SUPPORTED = True


def load(model=None):
    from f5_tts.api import F5TTS

    f5 = F5TTS(model=DEFAULT_MODEL, ckpt_file=model or "")
    ref_audio = str(files("f5_tts").joinpath(_REF_FILENAME))
    return f5, ref_audio, REF_TEXT


def synthesize(model, message: str, output_path: str) -> None:
    f5, ref_audio, ref_text = model

    wav, sr, _ = f5.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=message,
        show_info=lambda *a, **k: None,  # keep service output clean
        progress=None,  # safe: infer_process guards on None
    )

    audio = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    segment = AudioSegment(
        audio.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )
    segment.export(output_path, format="mp3")