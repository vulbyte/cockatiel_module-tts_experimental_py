"""
MeloTTS worker (MeloTTS, pip package; `from melo.api import TTS`). Default
language EN, speaker EN-Default.

MeloTTS is multilingual but a given TTS instance is built for exactly one
language, so this worker is fixed to EN. It writes wav via soundfile by
default; this worker instead passes output_path=None so tts_to_file returns
the concatenated float32 numpy array, then re-exports a real mp3 through
pydub. The speaker id is read from the TTS_SPEAKER_INDEX env var only when it
is a valid MeloTTS speaker id for the loaded language, otherwise it falls
back to EN-Default -- the service sets TTS_SPEAKER_INDEX to the SpeechT5 CMU
ARCTIC xvector index (default 7306), which is NOT a valid MeloTTS id (the EN
map is 0..4), so the fallback is what actually runs unless an operator picks
a small value.

No MODEL_SOURCE_SUPPORTED: the released package (MeloTTS 0.1.1) is
fixed-architecture -- its TTS.__init__(language, device) only accepts a
language and downloads weights from a hardcoded URL. (GitHub main adds
config_path/ckpt_path, but that is not on PyPI and a single model_source
still cannot map onto a config+checkpoint pair cleanly.)

Target: MeloTTS 0.1.1 (PyPI).
"""

import os

import numpy as np

from pydub import AudioSegment

# (pip package, importable name) pairs the service auto-installs when missing.
REQUIRED_PIP = [("MeloTTS", "melo.api")]

DEFAULT_LANGUAGE = "EN"
DEFAULT_SPEAKER_NAME = "EN-Default"
SPEAKER_INDEX_ENV = "TTS_SPEAKER_INDEX"


def load(model=None):
    from melo.api import TTS

    tts = TTS(language=DEFAULT_LANGUAGE, device="auto")
    spk2id = tts.hps.data.spk2id
    sample_rate = int(tts.hps.data.sampling_rate)
    speaker_id = _pick_speaker_id(spk2id)
    return tts, speaker_id, sample_rate


def _pick_speaker_id(spk2id):
    """Honor TTS_SPEAKER_INDEX only when it names a valid MeloTTS speaker."""
    valid_ids = set(spk2id.values())
    env_index = os.environ.get(SPEAKER_INDEX_ENV)
    if env_index is not None and env_index.isdigit() and int(env_index) in valid_ids:
        return int(env_index)
    return spk2id[DEFAULT_SPEAKER_NAME]


def synthesize(model, message: str, output_path: str) -> None:
    tts, speaker_id, sample_rate = model

    # output_path=None makes MeloTTS return the concatenated float32 numpy
    # array instead of writing a file, so we always export a real mp3.
    audio = tts.tts_to_file(
        message, speaker_id, output_path=None, speed=1.0, quiet=True
    )

    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    segment = AudioSegment(
        audio.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    segment.export(output_path, format="mp3")