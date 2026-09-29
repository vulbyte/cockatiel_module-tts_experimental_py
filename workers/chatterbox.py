"""
Chatterbox worker (ResembleAI/chatterbox via chatterbox-tts).

Chatterbox is a zero-shot voice-cloning model with a built-in default voice
(conds.pt ships in the model repo), so `generate(text)` works without an
audio prompt -- the service only passes text + output_path. `load(model=...)`
accepts a local checkpoint dir (containing ve.safetensors, t3_cfg.safetensors,
s3gen.safetensors, tokenizer.json, and conds.pt) via ChatterboxTTS.from_local;
without one it downloads the stock ResembleAI/chatterbox weights via
from_pretrained. The local-dir override is what a model_source provides.
Output is 24 kHz mono float32; normalized through numpy and exported as a
real mp3 via pydub.

Target: chatterbox-tts (current main, ChatterboxTTS.from_pretrained /
from_local + generate(text)).
"""

import numpy as np

from pydub import AudioSegment

# (pip package, importable name) pairs the service auto-installs when missing.
REQUIRED_PIP = [("chatterbox-tts", "chatterbox")]

# load() accepts a local checkpoint dir via model_source (from_local).
MODEL_SOURCE_SUPPORTED = True


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model=None):
    from chatterbox.tts import ChatterboxTTS

    device = _device()
    if model:
        # A local checkpoint dir override (model_source). Must contain the
        # four safetensors/tokenizer files (plus conds.pt for the default voice).
        return ChatterboxTTS.from_local(model, device)
    return ChatterboxTTS.from_pretrained(device)


def synthesize(model, message: str, output_path: str) -> None:
    wav = model.generate(message)

    # generate() returns a (1, N) torch tensor; the numpy fallback keeps this
    # independent of torch (a hard dep anyway) and testable with a stub.
    audio = wav
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()

    audio = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    segment = AudioSegment(
        audio.tobytes(),
        frame_rate=model.sr,
        sample_width=2,
        channels=1,
    )
    segment.export(output_path, format="mp3")