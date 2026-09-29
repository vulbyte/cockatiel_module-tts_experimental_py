"""
Contract tests for the TTS worker modules.

Verifies, WITHOUT downloading any model or requiring the optional TTS
runtimes:
  * every workers/*.py module exposes `load` and `synthesize` callables --
    the "all workers function the same way" guarantee;
  * each NEW worker's synthesize() writes a REAL mp3 when driven with a
    minimal stub model whose API the worker actually calls (the exercise
    reaches the pydub export), and the output begins with mp3 magic bytes
    (`ID3` or the `\\xff\\xfb` frame sync);
  * MODEL_SOURCE_SUPPORTED, when present, is a bool;
  * the hardened WorkerManager.supports_model_source treats a worker that
    cannot import as unavailable (False) instead of raising, so a broken
    optional runtime can never abort an unrelated message.

Run with the stdlib runner (no third-party deps beyond the module's own):

    python3 -m unittest test_workers -v
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

WORKERS_DIR = Path(__file__).parent / "workers"

MP3_MAGIC = (b"ID3", b"\xff\xfb")

# The six optional-runtime workers added alongside this test; each one's
# synthesize() reaches the pydub export using only the stub API it calls, so
# every one can be driven here. (The pre-existing workers -- mms, qwen3,
# speecht5, vibevoice -- are covered by the contract test above; their
# synthesize() shapes are heavier and out of scope for stub-driving.)
NEW_WORKERS = ["xtts", "piper", "kokoro", "melo", "f5", "chatterbox"]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"test_workers.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- Stub models: tiny fakes whose API each worker's synthesize() actually
# calls, so the pydub mp3 export is exercised without the real runtime. ---


def _xtts_stub_model():
    class _TTS:
        def tts(self, text, speaker=None, language=None):
            return __import__("numpy").zeros(2400, dtype="float32")

    return (_TTS(), "StubSpeaker")


def _piper_stub_model():
    class _Chunk:
        sample_rate = 22050
        sample_width = 2
        sample_channels = 1
        audio_float_array = __import__("numpy").zeros(22050, dtype="float32")

    class _Voice:
        def synthesize(self, text):
            yield _Chunk()

    return _Voice()


def _kokoro_stub_model():
    numpy = __import__("numpy")

    class _Result:
        audio = numpy.zeros(24000, dtype="float32")

    class _Pipeline:
        def __call__(self, text, voice=None):
            yield _Result()

    return (_Pipeline(), "af_heart")


def _melo_stub_model():
    class _TTS:
        def tts_to_file(self, text, speaker_id, output_path=None, speed=1.0, quiet=False):
            return __import__("numpy").zeros(44100, dtype="float32")

    return (_TTS(), 0, 44100)


def _f5_stub_model():
    class _F5:
        def infer(self, ref_file, ref_text, gen_text, show_info=None, progress=None):
            return __import__("numpy").zeros(24000, dtype="float32"), 24000, None

    return (_F5(), "/tmp/ref.wav", "some reference transcript")


def _chatterbox_stub_model():
    class _CB:
        sr = 24000

        def generate(self, text):
            return __import__("numpy").zeros(24000, dtype="float32")

    return _CB()


class WorkerContractTest(unittest.TestCase):
    """The 'all workers function the same way' guarantee."""

    def test_all_workers_expose_load_and_synthesize(self):
        workers = sorted(
            p.stem for p in WORKERS_DIR.glob("*.py") if not p.stem.startswith("_")
        )
        self.assertTrue(workers, "no worker modules found")
        for name in workers:
            with self.subTest(worker=name):
                module = _load_module(name, WORKERS_DIR / f"{name}.py")
                self.assertTrue(
                    callable(getattr(module, "load", None)), f"{name}.load"
                )
                self.assertTrue(
                    callable(getattr(module, "synthesize", None)),
                    f"{name}.synthesize",
                )

    def test_model_source_supported_is_bool_when_present(self):
        for p in WORKERS_DIR.glob("*.py"):
            name = p.stem
            if name.startswith("_"):
                continue
            with self.subTest(worker=name):
                module = _load_module(name, p)
                if hasattr(module, "MODEL_SOURCE_SUPPORTED"):
                    self.assertIsInstance(
                        module.MODEL_SOURCE_SUPPORTED, bool, name
                    )

    def test_model_source_supported_matches_the_worker_design(self):
        # Pin the truth table: a worker either accepts a model_source override
        # in load() or it does not. This caught a real regression where
        # chatterbox's load() accepted a local checkpoint dir (from_local) but
        # the flag was missing, so the service would never feed it a source.
        # A worker whose flag disagrees with its own load() is broken, not a
        # style choice.
        expected = {
            "chatterbox": True,  # from_local(checkpoint dir)
            "f5": True,          # ckpt_file= checkpoint path / hf:// URL
            "kokoro": True,      # KPipeline(repo_id=...)
            "melo": False,       # fixed-arch: TTS.__init__(language, device) only
            "mms": True,         # pipeline(model=...) generic
            "piper": True,       # PiperVoice.load(local .onnx)
            "qwen3": False,      # fixed fine-tune, no override
            "speecht5": False,   # fixed arch + fixed vocoder
            "vibevoice-1_5p": True,  # pipeline(model=...) generic
            "xtts": True,        # TTS(model_name=...) / TTS(model_path=, config_path=)
        }
        self.assertEqual(
            {p.stem for p in WORKERS_DIR.glob("*.py") if not p.stem.startswith("_")},
            set(expected),
            "the truth table must cover every worker (rename/extend the table, "
            "not just add a file)",
        )
        for name, want in expected.items():
            with self.subTest(worker=name):
                module = _load_module(name, WORKERS_DIR / f"{name}.py")
                self.assertEqual(
                    bool(getattr(module, "MODEL_SOURCE_SUPPORTED", False)),
                    want,
                    f"{name}: MODEL_SOURCE_SUPPORTED must be {want}",
                )


class WorkerMp3ExportTest(unittest.TestCase):
    """Each new worker's synthesize() writes a real mp3 via pydub."""

    STUBS = {
        "xtts": _xtts_stub_model,
        "piper": _piper_stub_model,
        "kokoro": _kokoro_stub_model,
        "melo": _melo_stub_model,
        "f5": _f5_stub_model,
        "chatterbox": _chatterbox_stub_model,
    }

    def _assert_writes_real_mp3(self, name, stub_model):
        module = _load_module(name, WORKERS_DIR / f"{name}.py")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.mp3"
            module.synthesize(stub_model, "Hello.", str(output))
            self.assertTrue(output.exists(), f"{name} wrote nothing to disk")
            data = output.read_bytes()
            self.assertGreater(len(data), 4, f"{name} output too small")
            self.assertTrue(
                data.startswith(MP3_MAGIC[0]) or data.startswith(MP3_MAGIC[1]),
                f"{name} output is not an mp3 (got {data[:16]!r})",
            )

    def test_each_worker_exports_a_real_mp3(self):
        for name, stub_factory in self.STUBS.items():
            with self.subTest(worker=name):
                self._assert_writes_real_mp3(name, stub_factory())


class WorkerManagerHardeningTest(unittest.TestCase):
    def test_broken_worker_is_supported_false_not_error(self):
        # A worker whose module cannot even import (e.g. a missing optional
        # runtime) must be reported as source-unsupported, never raise --
        # tts_service calls supports_model_source OUTSIDE its per-worker
        # try/except, so an exception here would abort a whole message.
        from worker_manager import WorkerManager

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            broken = tmp_dir / "broken_worker.py"
            broken.write_text(
                "import definitely_not_a_real_module_xyz\n"
                "def load(model=None):\n"
                "    return None\n"
                "def synthesize(model, message, output_path):\n"
                "    pass\n"
            )
            try:
                manager = WorkerManager(workers_dir=tmp_dir)
                self.assertFalse(manager.supports_model_source("broken_worker"))
            finally:
                broken.unlink(missing_ok=True)


class WorkerManagerWarmUpTest(unittest.TestCase):
    """warm_up() loads the model AND renders a clip, exactly like synthesize.

    The whole point of the startup gate is that warm_up runs BEFORE the module
    connects, so a worker that cannot load or render fails HERE instead of on
    the first real chat message. This must therefore behave identically to
    synthesize -- load once, render, write a real mp3.
    """

    def test_warm_up_loads_once_and_renders_a_real_mp3(self):
        from worker_manager import WorkerManager

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # A stub worker whose load() records how often it is called and
            # synthesize() writes a real mp3 magic-byte file.
            (tmp_dir / "stub.py").write_text(
                "import numpy as np\n"
                "from pydub import AudioSegment\n"
                "LOAD_CALLS = []\n"
                "def load(model=None):\n"
                "    LOAD_CALLS.append(model)\n"
                "    return object()\n"
                "def synthesize(model, message, output_path):\n"
                "    audio = np.zeros(8000, dtype='float32')\n"
                "    seg = AudioSegment(audio.tobytes(), frame_rate=8000, sample_width=2, channels=1)\n"
                "    seg.export(output_path, format='mp3')\n"
            )
            try:
                manager = WorkerManager(workers_dir=tmp_dir)
                module = manager._get_module("stub")
                out = tmp_dir / "warmup.mp3"

                manager.warm_up("stub", "hello, this is a message from cockatiel", str(out), "some/source")

                # Loaded exactly once, with the source override passed through.
                self.assertEqual(module.LOAD_CALLS, ["some/source"])
                # warm_up wrote a real mp3, not an empty placeholder.
                data = out.read_bytes()
                self.assertGreater(len(data), 4)
                self.assertTrue(
                    data.startswith(MP3_MAGIC[0]) or data.startswith(MP3_MAGIC[1]),
                    f"warm_up did not write an mp3 (got {data[:16]!r})",
                )
                # The model is cached: a second warm_up does NOT re-load.
                manager.warm_up("stub", "again", str(out), "some/source")
                self.assertEqual(module.LOAD_CALLS, ["some/source"], "model must be cached")
            finally:
                (tmp_dir / "stub.py").unlink(missing_ok=True)


class WorkerDependencyTest(unittest.TestCase):
    """Auto-install of a worker's runtime dependency.

    The operator should never be prompted to `pip install` by hand: the module
    installs a missing runtime before the warm-up, so a bad URL / missing
    package is fixed automatically instead of erroring the first message.
    """

    def test_required_pip_is_well_formed_on_every_worker(self):
        # Every worker with a runtime dependency declares (pip package, import
        # name) pairs; the import name is what find_spec checks, the pip name is
        # what gets installed. A worker that needs nothing declares nothing.
        for p in WORKERS_DIR.glob("*.py"):
            name = p.stem
            if name.startswith("_"):
                continue
            with self.subTest(worker=name):
                module = _load_module(name, p)
                deps = getattr(module, "REQUIRED_PIP", [])
                for pair in deps:
                    self.assertIsInstance(pair, tuple, name)
                    self.assertEqual(len(pair), 2, f"{name}: {pair}")
                    self.assertTrue(pair[0], f"{name}: empty pip package")
                    self.assertTrue(pair[1], f"{name}: empty import name")

    def test_ensure_dependencies_installs_only_missing_imports(self):
        from unittest import mock

        from worker_manager import WorkerManager

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Stub declares two deps: one already importable (numpy), one that
            # is definitely missing (a package that cannot exist).
            (tmp_dir / "depstub.py").write_text(
                "REQUIRED_PIP = [('numpy', 'numpy'), ('definitely-not-a-real-pkg-xyz', 'definitely_not_a_real_pkg_xyz')]\n"
                "def load(model=None):\n"
                "    return None\n"
                "def synthesize(model, message, output_path):\n"
                "    pass\n"
            )
            try:
                manager = WorkerManager(workers_dir=tmp_dir)

                calls = []

                def fake_run(cmd, **kwargs):
                    calls.append(cmd)
                    return mock.Mock(returncode=0, stderr="", stdout="ok")

                with mock.patch.object(manager, "_get_module", wraps=manager._get_module):
                    pass  # _get_module is fine; the stub imports cleanly

                import worker_manager as wm
                with mock.patch.object(wm.subprocess, "run", side_effect=fake_run):
                    installed = manager.ensure_dependencies("depstub")

                # numpy was present (no install); only the missing pkg installed.
                self.assertEqual(installed, ["definitely-not-a-real-pkg-xyz"])
                self.assertEqual(len(calls), 1, f"calls: {calls}")
                self.assertEqual(
                    calls[0][-1],
                    "definitely-not-a-real-pkg-xyz",
                    "the missing package is what pip installs",
                )
                # The command is the python interpreter's own pip, so the module
                # installs into the environment that is actually running it.
                self.assertIn("pip", calls[0])
            finally:
                (tmp_dir / "depstub.py").unlink(missing_ok=True)

    def test_ensure_dependencies_reports_without_installing_when_disabled(self):
        from unittest import mock

        import worker_manager as wm

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            (tmp_dir / "depstub2.py").write_text(
                "REQUIRED_PIP = [('definitely-not-a-real-pkg-abc', 'definitely_not_a_real_pkg_abc')]\n"
                "def load(model=None):\n"
                "    return None\n"
                "def synthesize(model, message, output_path):\n"
                "    pass\n"
            )
            try:
                manager = wm.WorkerManager(workers_dir=tmp_dir)
                with mock.patch.object(wm.subprocess, "run") as fake_run:
                    installed = manager.ensure_dependencies("depstub2", auto_install=False)
                self.assertEqual(installed, [], "auto_install=False must not install")
                fake_run.assert_not_called()
            finally:
                (tmp_dir / "depstub2.py").unlink(missing_ok=True)

    def test_ensure_torch_compat_detects_a_mismatch_and_fixes_it(self):
        from unittest import mock

        from worker_manager import WorkerManager

        manager = WorkerManager(workers_dir="workers")

        def fake_version(dist):
            # Simulate the broken pair: torch 2.6.0 with torchvision 0.27.0.
            return {"torch": "2.6.0", "torchvision": "0.27.0"}[dist]

        with mock.patch(
            "worker_manager.importlib.metadata.version", side_effect=fake_version
        ):
            report = manager.ensure_torch_compat(auto_install=False)
            self.assertIn("MISMATCH", report, report)
            self.assertIn("0.21.0", report, report)

        # auto_install=True runs pip with the corrected torchvision.
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return mock.Mock(returncode=0, stderr="", stdout="ok")

        import worker_manager as wm

        with mock.patch(
            "worker_manager.importlib.metadata.version", side_effect=fake_version
        ), mock.patch.object(wm.subprocess, "run", side_effect=fake_run):
            report = manager.ensure_torch_compat(auto_install=True)
        self.assertIn("aligned torchvision", report, report)
        self.assertEqual(len(calls), 1, f"calls: {calls}")
        self.assertIn("torchvision==0.21.0", calls[0][-1], calls)

    def test_ensure_torch_compat_passes_a_matching_pair(self):
        from unittest import mock

        from worker_manager import WorkerManager

        manager = WorkerManager(workers_dir="workers")

        def fake_version(dist):
            return {"torch": "2.6.0", "torchvision": "0.21.0"}[dist]

        with mock.patch(
            "worker_manager.importlib.metadata.version", side_effect=fake_version
        ):
            report = manager.ensure_torch_compat(auto_install=False)
            self.assertIn("match", report, report)


if __name__ == "__main__":
    unittest.main()