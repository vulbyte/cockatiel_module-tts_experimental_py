"""
Dynamically loads and caches TTS "worker" modules from a folder.

Each worker module (a .py file dropped into WORKERS_DIR) must expose:

    def load() -> Any:
        Load and return whatever model/pipeline/tokenizer object(s) the
        worker needs. Called once per worker, the first time it's used,
        and the result is cached and reused for every later synthesize()
        call for the rest of the process's life -- these models are far
        too slow to load fresh for every chat message.

    def synthesize(model: Any, message: str, output_path: str) -> None:
        Run inference with the already-loaded `model` and write the
        resulting audio to `output_path`.

See workers/mms.py, workers/qwen3.py, workers/speecht5.py, and
workers/vibevoice.py for reference implementations.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger("worker_manager")


class WorkerError(Exception):
    """Raised when a worker can't be found, loaded, or fails during synthesis."""


class WorkerManager:
    def __init__(self, workers_dir: Union[Path, str] = "workers"):
        self.workers_dir = Path(workers_dir)
        self._modules: Dict[str, ModuleType] = {}
        self._loaded_models: Dict[str, Any] = {}
        # Guards model loading so two near-simultaneous requests for a
        # not-yet-loaded model don't both trigger a (very expensive) load.
        self._load_lock = threading.Lock()

    def available_workers(self) -> List[str]:
        """Names (file stems) of every worker module found in workers_dir."""
        if not self.workers_dir.is_dir():
            return []
        return sorted(
            p.stem
            for p in self.workers_dir.glob("*.py")
            if not p.stem.startswith("_")
        )

    def _get_module(self, name: str) -> ModuleType:
        if name in self._modules:
            return self._modules[name]

        path = self.workers_dir / f"{name}.py"
        if not path.exists():
            available = ", ".join(self.available_workers()) or "(none found)"
            raise WorkerError(
                f"No worker named '{name}' in {self.workers_dir}/. "
                f"Available: {available}"
            )

        spec = importlib.util.spec_from_file_location(f"workers.{name}", path)
        if spec is None or spec.loader is None:
            raise WorkerError(f"Could not load worker module from {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        for required in ("load", "synthesize"):
            if not hasattr(module, required):
                raise WorkerError(
                    f"Worker '{name}' is missing a required `{required}()` function"
                )

        self._modules[name] = module
        return module

    def _get_model(self, name: str, model_source: str = "") -> Any:
        # The model cache is keyed by worker + source so a custom HF id / local
        # path switches cleanly instead of reusing a stale cached model.
        key = f"{name}|{model_source}" if model_source else name
        if key in self._loaded_models:
            return self._loaded_models[key]

        with self._load_lock:
            # Another thread may have finished loading while we waited.
            if key in self._loaded_models:
                return self._loaded_models[key]

            module = self._get_module(name)
            logger.info("Loading worker model '%s' (first use)...", name)
            model = module.load(model_source or None)
            self._loaded_models[key] = model
            logger.info("Worker model '%s' loaded.", name)
            return model

    def synthesize(self, name: str, message: str, output_path: str, model_source: str = "") -> None:
        """Blocking call -- run this in a thread/executor from async code."""
        module = self._get_module(name)
        model = self._get_model(name, model_source)
        module.synthesize(model, message, output_path)

    def warm_up(self, name: str, message: str, output_path: str, model_source: str = "") -> None:
        """Load the model AND render one clip, proving the worker works end to end.

        Identical to `synthesize` -- the point is the call site: this is used
        at STARTUP, before the module connects to the engine, so a worker that
        cannot load its model or render audio fails here instead of on the
        first real chat message (which used to be when the model was lazily
        loaded, erroring that message).
        """
        self.synthesize(name, message, output_path, model_source)

    def supports_model_source(self, name: str) -> bool:
        """Whether a worker's load() accepts a `model_source` override.

        A worker declares support by setting a module-level
        `MODEL_SOURCE_SUPPORTED = True` (see workers/mms.py). Workers with a
        fixed architecture (SpeechT5, Qwen3) don't declare it, so a custom HF
        id / local dir configured for another model is never force-fed into
        them.
        """
        try:
            module = self._get_module(name)
        except Exception:
            # A worker whose module can't even be imported (e.g. a missing
            # optional runtime imported at top level) is treated as having no
            # source support rather than raising: tts_service calls this OUTSIDE
            # its per-worker try/except, so an import error here would otherwise
            # abort the whole fallback chain for a message.
            logger.warning(
                "Worker '%s' could not be imported; treating as source-unsupported.",
                name,
                exc_info=True,
            )
            return False
        return bool(getattr(module, "MODEL_SOURCE_SUPPORTED", False))

    def required_pip(self, name: str) -> List[Tuple[str, str]]:
        """The (pip package, importable name) pairs a worker needs.

        Declared as a module-level `REQUIRED_PIP` list on the worker. Returns
        an empty list for a worker that declares none (no runtime dependency
        beyond the hard deps).
        """
        module = self._get_module(name)
        raw = getattr(module, "REQUIRED_PIP", [])
        return [(str(p), str(i)) for p, i in raw]

    def ensure_dependencies(
        self,
        name: str,
        auto_install: bool = True,
        pip: List[str] = None,
    ) -> List[str]:
        """Make sure every runtime a worker needs is importable.

        Returns the pip packages that were INSTALLED (empty if all were already
        present). With `auto_install=False` it only reports what is missing and
        installs nothing, so a headless check or a dry run can call it without
        touching the environment. `pip` is the install command prefix (default
        `[sys.executable, "-m", "pip", "install"]`), injectable for tests.

        The lazy-import design means the worker's module imports cleanly even
        when its runtime is absent, so the missing import can be detected here
        (via importlib.util.find_spec) and installed BEFORE load() runs. A
        worker whose runtime install fails is still skipped by the service's
        fallback chain -- this just makes the happy path work without the
        operator reaching for a shell.
        """
        installed: List[str] = []
        for pip_pkg, import_name in self.required_pip(name):
            if importlib.util.find_spec(import_name) is not None:
                continue
            if not auto_install:
                logger.warning(
                    "Worker '%s' needs '%s' (import '%s') which is not installed.",
                    name, pip_pkg, import_name,
                )
                continue
            logger.info(
                "Worker '%s' needs '%s' (import '%s') — installing...",
                name, pip_pkg, import_name,
            )
            cmd = (pip or [sys.executable, "-m", "pip", "install"]) + [pip_pkg]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except Exception as e:
                logger.error(
                    "Auto-install of '%s' for worker '%s' failed to run: %s",
                    pip_pkg, name, e,
                )
                continue
            if result.returncode != 0:
                logger.error(
                    "Auto-install of '%s' for worker '%s' FAILED (exit %d):\n%s",
                    pip_pkg, name, result.returncode, result.stderr.strip() or result.stdout.strip(),
                )
                continue
            logger.info("Installed '%s' for worker '%s'.", pip_pkg, name)
            installed.append(pip_pkg)
        return installed

    def unload(self, name: str, model_source: str = "") -> None:
        """Drop a cached model (e.g. to free GPU memory before switching).

        Keyed on the same `name|source` composite used by `_get_model`, so a
        source-parameterized model is actually released from the cache.
        """
        key = f"{name}|{model_source}" if model_source else name
        self._loaded_models.pop(key, None)

    def ensure_torch_compat(self, auto_install: bool = True) -> str:
        """Detect (and optionally fix) a torch / torchvision version mismatch.

        PyTorch pairs each `torch` release with a specific `torchvision` (the
        rule: torchvision's minor is torch's minor + 15, e.g. torch 2.6.0 ↔
        torchvision 0.21.0). A mismatched pair makes `import torchvision`
        crash with `RuntimeError: operator torchvision::nms does not exist`,
        which in turn breaks `transformers` — every torch-based TTS worker
        then fails to load and the service sits in the TUI's "starting" loop
        forever. This is a broken ENVIRONMENT, not a worker bug, so it is
        fixed here rather than surfaced as an opaque per-worker error.

        Reads versions via `importlib.metadata` (metadata only, never imports
        torch — importing torch is exactly what crashes on a mismatch). Returns
        a human-readable result describing what was found and what was done.
        """
        try:
            import importlib.metadata as md

            torch_ver = md.version("torch")
            torchvision_ver = md.version("torchvision")
        except md.PackageNotFoundError:
            # torch/torchvision absent entirely: a worker that needs it simply
            # won't be importable, and `ensure_dependencies` handles that.
            return "torch not installed; nothing to align"
        except Exception as e:  # noqa: BLE001 -- metadata reads must never kill warm-up
            return f"could not read torch versions: {e}"

        def minor(v: str) -> int:
            try:
                return int(v.split(".")[1])
            except (IndexError, ValueError):
                return -1

        torch_minor = minor(torch_ver)
        tv_minor = minor(torchvision_ver)
        expected_tv = f"0.{torch_minor + 15}.0" if torch_minor >= 0 else None

        if expected_tv and tv_minor == torch_minor + 15:
            return (
                f"torch {torch_ver} + torchvision {torchvision_ver} "
                "match; no action needed"
            )
        if expected_tv is None:
            return f"unparseable torch version {torch_ver!r}; not aligning"

        if not auto_install:
            return (
                f"MISMATCH: torch {torch_ver} pairs with torchvision "
                f"{expected_tv}, but torchvision {torchvision_ver} is installed. "
                "Run with auto_install_deps on (or pip install "
                f"'torchvision=={expected_tv}') to fix."
            )

        logger.warning(
            "torch/torchvision MISMATCH: torch %s needs torchvision %s, but "
            "%s is installed — installing the matching torchvision. (A broken "
            "pair crashes import torchvision and takes every torch-based worker "
            "down with it.)",
            torch_ver, expected_tv, torchvision_ver,
        )
        cmd = [
            sys.executable, "-m", "pip", "install", "--upgrade",
            f"torchvision=={expected_tv}",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except Exception as e:
            return f"auto-fix failed to run: {e}"
        if result.returncode != 0:
            return (
                f"auto-fix FAILED (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return (
            f"aligned torchvision {torchvision_ver} -> {expected_tv} "
            f"to match torch {torch_ver}"
        )
