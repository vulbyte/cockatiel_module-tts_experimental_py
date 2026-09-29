"""
XTTS worker (coqui-tts: tts_models/multilingual/multi-dataset/xtts_v2).

Coqui XTTS-v2 is a multilingual voice-cloning model and requires a reference
voice: either a cloned `speaker_wav` or a built-in preset speaker from the
model's speakers_xtts.pth. The service only passes text + output_path, so
this worker picks a preset speaker at load time from `tts.speakers`,
preferring the documented built-in "Craig Gutsy" and otherwise falling back
to the first available speaker. `load(model=...)` accepts a model name (e.g.
another `tts_models/...` id) or a local checkpoint directory containing
model.pth + config.json, so a model_source override is supported.

Note: coqui-ai/TTS is archived; `coqui-tts` (idiap fork) is the maintained
package. Target: coqui-tts (current main / 0.2x), which also exposes
`TTS(model_path=..., config_path=...)` for local checkpoints.
"""

import numpy as np
from pathlib import Path

from pydub import AudioSegment

# (pip package, importable name) pairs the service auto-installs when missing.
REQUIRED_PIP = [("coqui-tts", "TTS")]

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000  # XTTS-v2's native output rate
LANGUAGE = "en"
# A built-in preset speaker from XTTS-v2's speakers_xtts.pth. Used when the
# service doesn't pass a reference wav; some builds may not ship this exact
# name, so load() falls back to tts.speakers[0].
DEFAULT_SPEAKER = "Craig Gutsy"

# load() accepts a checkpoint dir or a model name via model_source.
MODEL_SOURCE_SUPPORTED = True


def load(model=None):
    import torch

    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model:
        model_dir = Path(model)
        if model_dir.is_dir():
            # A local XTTS checkpoint dir (model.pth + config.json + vocab.json
            # + speakers_xtts.pth). The API loads the config + checkpoint pair.
            tts = TTS(
                model_path=str(model_dir / "model.pth"),
                config_path=str(model_dir / "config.json"),
            ).to(device)
        else:
            # A named model, e.g. a different tts_models/... id.
            tts = TTS(model_name=model).to(device)
    else:
        tts = TTS(model_name=MODEL_NAME).to(device)

    # Pick the preset voice now so synthesize() only needs text + output_path.
    speakers = tts.speakers or []
    speaker = (
        DEFAULT_SPEAKER
        if DEFAULT_SPEAKER in speakers
        else (speakers[0] if speakers else None)
    )
    if speaker is None:
        raise RuntimeError(
            "XTTS loaded with no preset speakers; a speaker_wav is required "
            "but the service only passes text."
        )
    return tts, speaker


def synthesize(model, message: str, output_path: str) -> None:
    tts, speaker = model
    wav = tts.tts(text=message, speaker=speaker, language=LANGUAGE)

    audio = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    segment = AudioSegment(
        audio.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1,
    )
    segment.export(output_path, format="mp3")