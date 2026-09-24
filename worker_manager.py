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
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Union

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

    def supports_model_source(self, name: str) -> bool:
        """Whether a worker's load() accepts a `model_source` override.

        A worker declares support by setting a module-level
        `MODEL_SOURCE_SUPPORTED = True` (see workers/mms.py). Workers with a
        fixed architecture (SpeechT5, Qwen3) don't declare it, so a custom HF
        id / local dir configured for another model is never force-fed into
        them.
        """
        module = self._get_module(name)
        return bool(getattr(module, "MODEL_SOURCE_SUPPORTED", False))

    def unload(self, name: str, model_source: str = "") -> None:
        """Drop a cached model (e.g. to free GPU memory before switching).

        Keyed on the same `name|source` composite used by `_get_model`, so a
        source-parameterized model is actually released from the cache.
        """
        key = f"{name}|{model_source}" if model_source else name
        self._loaded_models.pop(key, None)
