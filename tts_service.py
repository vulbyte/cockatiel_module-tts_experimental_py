"""
tts_service.py
Cockatiel TTS Module Entrypoint & Service Integration
"""

import os
import sys
from pathlib import Path

# 1. The python client is vendored alongside this module (self-contained; the
# module no longer reaches into the engine's source tree). The vendored proto
# is resolved by the client next to its own file.
import os
import sys
from pathlib import Path

# Now import safely
from cockatiel_client import CockatielClient, pb

import argparse
import asyncio
import atexit
import concurrent.futures
import json
import logging
import math
import time
from collections import OrderedDict

from worker_manager import WorkerManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("tts_service")

CONFIG_PATH = Path("config.json")
CLIPS_DIR = Path("clips")


def get_safe_filename(text: str, max_length: int = 50) -> str:
    """Replaces spaces with underscores, removes invalid chars, and trims length."""
    sanitized = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text.replace(" ", "_"))
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")[:max_length]


def fix_ssl_certs() -> None:
    """Point Python's default SSL context at a real CA bundle.

    The python.org macOS installer ships WITHOUT a usable cert.pem (its
    `Install Certificates.command` is never run), so `import ssl` uses an
    empty bundle and EVERY https download fails with
    `SSL: CERTIFICATE_VERIFY_FAILED` — which would otherwise disable piper's
    voice download, HF model downloads, and everything else that touches the
    network. certifi (bundled with most stacks) or the system bundle both
    work; whichever is present is used. This runs once at startup, before any
    worker loads, so the module "just works" on a machine whose Python was
    never cert-fixed.
    """
    import ssl

    candidates = []
    try:
        import certifi

        candidates.append(certifi.where())
    except Exception:  # noqa: BLE001 -- a missing certifi is fine, not fatal
        pass
    for system in ("/etc/ssl/cert.pem", "/etc/pki/tls/certs/ca-bundle.crt"):
        if os.path.exists(system):
            candidates.append(system)

    for cafile in candidates:
        try:
            ctx = ssl.create_default_context(cafile=cafile)
            # Sanity-check the bundle actually verifies, so a corrupt file is
            # skipped rather than replacing one failure with another.
            ctx.verify_mode = ssl.CERT_REQUIRED
        except Exception:  # noqa: BLE001
            continue
        ssl._create_default_https_context = lambda: ctx  # noqa: SLF001
        if os.environ.get("SSL_CERT_FILE") != cafile:
            os.environ["SSL_CERT_FILE"] = cafile
        logger.info("SSL: using CA bundle %s", cafile)
        return

    logger.warning("SSL: no usable CA bundle found; https downloads may fail")


def resolve_model_source(source: str) -> str:
    """Normalize a model source into a form `pipeline(model=...)` can load.

    Accepts a bare HF repo id ("facebook/mms-1b-all"), a local directory path,
    or a HuggingFace model URL such as
    "https://huggingface.co/facebook/mms-1b-all". A trailing "?", query string,
    or page path ("/tree/main") is tolerated and stripped; the URL is reduced
    to its "<org>/<model>" repo id, which is what the workers pass to
    `pipeline(model=...)`. Anything that is not an http(s) URL is passed
    through untouched (repo id or local dir, as before).
    """
    source = (source or "").strip()
    if not source or not source.startswith(("http://", "https://")):
        return source
    # Drop query/fragment and any trailing slash before splitting the path.
    no_query = source.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    # "https://<host>/<org>/<model>(/anything-else)" -> <org>/<model>.
    parts = no_query.split("://", 1)[-1].split("/", 2)
    if len(parts) >= 3:
        return f"{parts[1]}/{parts[2].split('/', 1)[0]}"
    return no_query


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cockatiel TTS Module")
    parser.add_argument("--ip", type=str, help="Engine IP address")
    parser.add_argument("-p", "--port", type=int, help="Engine WebSocket port")
    parser.add_argument("--model", type=str, help="Default TTS worker model name")
    parser.add_argument("--test", type=str, help="Test TTS synthesis locally with a given message without connecting to Cockatiel")
    parser.add_argument("-n", "--new", action="store_true", help="Reset configuration and regenerate defaults on next start")
    parser.add_argument("--pin", type=str, help="Engine pairing PIN (accepted for CLI compatibility; config file takes precedence)")
    parser.add_argument("--name", type=str, help="Module name override")
    return parser.parse_args()


def load_env_file(path: str):
    """Load a KEY=VALUE `.env` file into the environment (real env wins)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def load_or_setup_config(args) -> dict:
    if args.new and CONFIG_PATH.exists():
        logger.info("--new flag detected. Removing old config.json...")
        try:
            CONFIG_PATH.unlink()
        except OSError as e:
            logger.error("Failed to delete config.json: %s", e)

    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            config = {}

    if not config:
        logger.warning(
            "No config.json found. Writing default config "
            "(engine 127.0.0.1:9734). Address is overridden by --ip/--port/--pin."
        )
        config = {
            "engine_ip": "127.0.0.1",
            "engine_port": 9734,
            "model": "piper",
        }
        CONFIG_PATH.write_text(json.dumps(config, indent=2))

    # Settings only — the pairing PIN is a secret and lives in the
    # environment (COCKATIEL_PIN / --pin), never in config.json.
    if args.ip:
        config["engine_ip"] = args.ip
    if args.port:
        config["engine_port"] = args.port
    if args.model:
        config["model"] = args.model

    config.setdefault("volume", 0.4)
    config.setdefault("play_locally", False)
    # Optional per-worker model override: an HF model id, a HuggingFace model
    # URL (e.g. "https://huggingface.co/facebook/mms-1b-all", normalized to its
    # repo id), OR a local model dir — passed to the active worker's
    # load(model=...). Setting stays in config.
    config.setdefault("model_source", "")
    # Safety caps so a pasted novel can't spawn a multi-minute render or push a
    # giant blob into the timeline DB. All config-driven with sane defaults.
    config.setdefault("max_chars", 1000)
    config.setdefault("max_audio_bytes", 5 * 1024 * 1024)
    config.setdefault("inference_timeout", 60)
    # Tuning values (all defaulted; created in config.json when missing).
    # How many syntheses may run concurrently (1 = fully serialized).
    config.setdefault("synthesis_concurrency", 1)
    # Cap on the recently-synthesized-uuid dedup set (oldest evicted).
    config.setdefault("max_dedup", 1000)
    # WebSocket open handshake timeout (seconds), passed to the client.
    config.setdefault("open_timeout", 10)
    # Engine auth handshake reply timeout (seconds), passed to the client.
    config.setdefault("handshake_timeout", 10)
    # Cap on in-flight background handler tasks in the client.
    config.setdefault("max_pending_tasks", 16)
    # Reconnect backoff base (seconds).
    config.setdefault("reconnect_backoff_base", 1.0)
    # Reconnect backoff cap (seconds).
    config.setdefault("reconnect_backoff_max", 30.0)
    # A session this long (seconds) resets the reconnect backoff to base.
    config.setdefault("backoff_reset_threshold", 30)
    # Dedicated thread pool for synthesis/playback; 0 = default executor.
    config.setdefault("worker_threads", 0)
    # Auto-install a worker's missing runtime dependency (its REQUIRED_PIP)
    # before the startup warm-up, so the operator is never prompted to pip
    # install by hand. A missing package used to surface as an error on the
    # first message; with this on, the module installs it and only reports
    # "starting" until the warm-up proves the worker renders. Set false to
    # refuse touching the environment.
    config.setdefault("auto_install_deps", True)
    # SpeechT5 CMU ARCTIC xvector index (voice); exported via TTS_SPEAKER_INDEX.
    config.setdefault("speaker_index", 7306)
    # Persist so the new setting always exists in config.json.
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    return config


def play_audio(path: str, volume: float = 1.0):
    """Play a rendered audio file at the given volume (0.0–1.0) using pydub + simpleaudio."""
    try:
        from pydub import AudioSegment
        import simpleaudio as sa

        segment = AudioSegment.from_file(path)
        if volume < 0.0:
            volume = 0.0
        if volume > 1.0:
            volume = 1.0
        # Apply volume as a dB change relative to full scale.
        db = 20.0 * math.log10(volume) if volume > 0.0 else -120.0
        segment = segment.apply_gain(db)
        play_obj = sa.play_buffer(
            segment.raw_data,
            num_channels=segment.channels,
            bytes_per_sample=segment.sample_width,
            sample_rate=segment.frame_rate,
        )
        play_obj.wait_done()
    except ImportError as e:
        logger.warning("Playback disabled (missing pydub/simpleaudio): %s", e)
    except Exception as e:
        logger.error("Playback failed: %s", e)


async def main():
    args = parse_arguments()
    # Fix the broken default SSL bundle BEFORE any worker downloads a model or
    # voice — otherwise every https fetch fails and the module is stuck in
    # "starting". Runs even before config load so the cert is ready regardless.
    fix_ssl_certs()
    config = load_or_setup_config(args)

    CLIPS_DIR.mkdir(exist_ok=True)

    manager = WorkerManager(workers_dir="workers")
    available = manager.available_workers()
    logger.info("Discovered local TTS workers: %s", available)

    if not available:
        logger.error("No workers found in 'workers/' folder!")
        return

    active_model = config.get("model", "piper")
    model_source = resolve_model_source(config.get("model_source", "") or "")

    # The `model` field names the worker (e.g. "mms"). A URL in that field
    # means "use whatever model this points at" — there is no worker named
    # after a URL, so fall back to the generic text-to-speech worker and feed
    # the URL in as the model source. Same end result as setting model_source
    # directly, but lets an operator paste a model link in one place.
    if active_model.startswith(("http://", "https://")):
        model_source = resolve_model_source(active_model)
        active_model = "mms"
        logger.info(
            "model is a URL — using the '%s' worker with source '%s'",
            active_model, model_source,
        )

    if active_model not in available:
        logger.warning("Configured model '%s' not found. Falling back to '%s'", active_model, available[0])
        active_model = available[0]

    # Optional dedicated executor for the blocking synthesize/playback calls;
    # 0/None keeps the asyncio default executor (behavior identical).
    worker_threads = int(config.get("worker_threads", 0) or 0)
    thread_executor = None
    if worker_threads > 0:
        thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_threads)
        atexit.register(thread_executor.shutdown)

    # Handle local testing mode via --test flag
    if args.test:
        logger.info("Running in TEST mode using model '%s'", active_model)
        logger.info("Test message: '%s'", args.test)
        
        # Auto-install the runtime like the warm-up path would, so --test is
        # not a different environment than a real run.
        if config.get("auto_install_deps", True):
            installed = manager.ensure_dependencies(active_model)
            if installed:
                logger.info(
                    "Auto-installed runtime package(s) %s for worker '%s'.",
                    installed, active_model,
                )
        safe_name = get_safe_filename(args.test)
        output_path = CLIPS_DIR / f"{safe_name}.mp3"

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                thread_executor,
                manager.synthesize,
                active_model,
                args.test,
                str(output_path),
                model_source,
            )
            logger.info("Test synthesis success! Audio saved to: %s", output_path.resolve())
        except Exception as e:
            logger.error("Test synthesis failed: %s", e)
        return

    # Normal Cockatiel Engine connection loop
    logger.info("TTS Service active using model worker: '%s'", active_model)

    # ── Startup warm-up gate ───────────────────────────────────────────────
    # A worker's model is loaded lazily, on the FIRST synthesis. If that model
    # is broken (bad URL, missing runtime, download failure), the error used to
    # land on the first real chat message. Worse, the module connected to the
    # engine the moment it started, so the engine routed messages to a worker
    # that was not actually ready.
    #
    # Fix: prove the active worker can load its model AND render a real clip
    # BEFORE connecting. While this runs the TUI shows the module as
    # "starting" (the supervisor set that flag at launch; the engine only
    # reports the module as connected once this process completes the
    # handshake), and the engine never routes a message to a worker that is
    # not ready. On failure we retry with backoff and stay unconnected, so a
    # transient download or a model that needs a moment to warm up does not
    # kill the module -- it just keeps showing "starting".
    inference_timeout = int(config.get("inference_timeout", 60))
    WARMUP_TEXT = "hello, this is a message from cockatiel"
    warmup_backoff = float(config.get("reconnect_backoff_base", 1.0))
    warmup_path = CLIPS_DIR / "startup_selftest.mp3"

    # Environment pre-flight: a torch/torchvision mismatch crashes import
    # torchvision (and with it transformers, and with it every torch-based
    # worker), which would otherwise make the warm-up loop below fail forever
    # and leave the module stuck in "starting". Detect + auto-fix it once,
    # before any worker tries to load. This is the automated version of what a
    # human would otherwise have to debug by hand.
    if config.get("auto_install_deps", True):
        compat_result = manager.ensure_torch_compat(auto_install=True)
    else:
        compat_result = manager.ensure_torch_compat(auto_install=False)
    if compat_result:
        logger.info("torch compat: %s", compat_result)
    while True:
        # Same candidate chain the live path uses: the configured worker first,
        # then every other available worker. A worker that is incompatible with
        # the configured model_source is skipped, exactly like the live loop.
        candidates = [active_model] + [
            w for w in available if w != active_model
        ]
        warmup_ok = False
        for worker in candidates:
            supports_source = manager.supports_model_source(worker)
            if model_source and not supports_source:
                continue
            worker_source = model_source if supports_source else ""
            # Auto-install the worker's runtime if it is missing, so the
            # warm-up below is not the first place a missing package surfaces
            # as an error. gated by auto_install_deps (default on).
            if config.get("auto_install_deps", True):
                installed = manager.ensure_dependencies(worker)
                if installed:
                    logger.info(
                        "Auto-installed runtime package(s) %s for worker '%s'.",
                        installed, worker,
                    )
            try:
                logger.info(
                    "Warm-up: rendering a test clip via worker '%s' (this loads "
                    "the model; a large one can take a while)...",
                    worker,
                )
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(
                        thread_executor,
                        manager.warm_up,
                        worker,
                        WARMUP_TEXT,
                        str(warmup_path),
                        worker_source,
                    ),
                    timeout=inference_timeout,
                )
                if warmup_path.exists() and warmup_path.stat().st_size > 0:
                    warmup_ok = True
                    logger.info(
                        "Warm-up OK: worker '%s' rendered a %d-byte test clip.",
                        worker,
                        warmup_path.stat().st_size,
                    )
                    break
                logger.error(
                    "Warm-up: worker '%s' wrote no audio; trying the next worker.",
                    worker,
                )
            except Exception as e:
                logger.error(
                    "Warm-up: worker '%s' failed to load or render: %s",
                    worker,
                    e,
                )
                continue
        if warmup_ok:
            break
        logger.error(
            "TTS warm-up FAILED for every worker — staying unconnected "
            "(the TUI shows 'starting'); retrying in %.0fs. Check the model "
            "source and the worker's own logs above (auto-install of missing "
            "runtimes is on unless auto_install_deps is false).",
            warmup_backoff,
        )
        await asyncio.sleep(warmup_backoff)
        warmup_backoff = min(warmup_backoff * 2, float(config.get("reconnect_backoff_max", 30.0)))

    # Modern config pattern: settings in config.json, secrets in .env / env.
    load_env_file(".env")
    engine_ip = config.get("engine_ip", "127.0.0.1")
    engine_port = int(config.get("engine_port", 9734))
    pairing_pin = int(os.environ.get("COCKATIEL_PIN") or args.pin or 0)
    module_name = args.name or "tts-experimental-py"

    # Serialize synthesis: the local model isn't safe for concurrent inference,
    # and handlers now run as background tasks so probes are never blocked. A
    # per-attempt deadline keeps a hung worker from holding the lock forever.
    # Concurrency is configurable (default 1 = fully serialized).
    synthesis_concurrency = max(1, int(config.get("synthesis_concurrency", 1)))
    synthesis_lock = asyncio.Semaphore(synthesis_concurrency)

    # Bounded dedup of recently-synthesized message uuids (oldest evicted) so a
    # re-delivered message isn't re-rendered a second time.
    max_dedup = int(config.get("max_dedup", 1000))
    recent_synthesis: OrderedDict[str, None] = OrderedDict()

    def mark_synthesized(uuid7: str) -> None:
        recent_synthesis[uuid7] = None
        while len(recent_synthesis) > max_dedup:
            recent_synthesis.popitem(last=False)

    max_chars = int(config.get("max_chars", 1000))
    max_audio_bytes = int(config.get("max_audio_bytes", 5 * 1024 * 1024))

    # Export the SpeechT5 speaker voice (CMU ARCTIC xvector index) to the
    # worker via env var — worker modules read it at model-load time.
    os.environ["TTS_SPEAKER_INDEX"] = str(int(config.get("speaker_index", 7306)))

    async def handle_synthesis(client, msg, container) -> None:
        """Render speech for one post-process message (runs as a background task)."""
        text_to_speak = msg.processed_message or ""
        if not text_to_speak.strip():
            if msg.raw_message is not None and msg.raw_message.raw_message:
                text_to_speak = msg.raw_message.raw_message
        if not text_to_speak.strip():
            return

        if len(text_to_speak) > max_chars:
            logger.info(
                "[TTS Engine] Truncating %d-char message to %d chars for %s",
                len(text_to_speak), max_chars, msg.message_uuid7,
            )
            text_to_speak = text_to_speak[:max_chars]

        # Skip a message we already rendered (engine re-delivery, e.g. after a
        # reconnect). Re-ack with the cached clip on disk so the stage still
        # completes, but never re-run inference for the same uuid.
        if msg.message_uuid7 in recent_synthesis:
            logger.info(
                "[TTS Engine] Skipping duplicate synthesis for %s (already rendered).",
                msg.message_uuid7,
            )
            reply = pb.MessagePostProcess(
                message_uuid7=msg.message_uuid7,
                processed_message=text_to_speak,
            )
            cached_path = CLIPS_DIR / f"{get_safe_filename(text_to_speak)}_{msg.message_uuid7[:8]}.mp3"
            if cached_path.exists():
                try:
                    reply.audio = cached_path.read_bytes()
                    reply.audio_type = "audio/mpeg"
                except OSError:
                    pass
            await client.send("message_post_process", reply)
            return

        logger.info("[TTS Engine] Rendering speech for message: '%s'", text_to_speak)
        safe_name = get_safe_filename(text_to_speak)
        output_path = CLIPS_DIR / f"{safe_name}_{msg.message_uuid7[:8]}.mp3"

        # Fallback order: the configured worker first, then every other
        # available worker. A worker that can't render (missing model, bad
        # source, runtime error) is skipped, not fatal. model_source is only
        # passed to workers that support it (see
        # WorkerManager.supports_model_source); workers incompatible with the
        # configured source are skipped, not errored.
        candidates = [active_model] + [
            w for w in available if w != active_model
        ]

        audio_bytes = b""
        rendered_by = None
        for worker in candidates:
            supports_source = manager.supports_model_source(worker)
            if model_source and not supports_source:
                logger.info(
                    "[TTS Engine] Skipping worker '%s': incompatible with configured model_source.",
                    worker,
                )
                continue
            worker_source = model_source if supports_source else ""
            try:
                loop = asyncio.get_running_loop()
                async with synthesis_lock:
                    await asyncio.wait_for(
                        loop.run_in_executor(
                            thread_executor,
                            manager.synthesize,
                            worker,
                            text_to_speak,
                            str(output_path),
                            worker_source,
                        ),
                        timeout=inference_timeout,
                    )
                audio_bytes = output_path.read_bytes()
                if len(audio_bytes) > max_audio_bytes:
                    logger.warning(
                        "[TTS Engine] Worker '%s' produced %d bytes for %s — "
                        "over %d-byte cap; dropping audio.",
                        worker, len(audio_bytes), msg.message_uuid7, max_audio_bytes,
                    )
                    audio_bytes = b""
                rendered_by = worker
                break
            except Exception as e:
                logger.error(
                    "[TTS Engine] Worker '%s' failed to render: %s",
                    worker,
                    e,
                )
                continue

        if rendered_by is None:
            logger.error(
                "[TTS Engine] All workers failed for %s — replying with no "
                "audio so the message completes instead of hanging.",
                msg.message_uuid7,
            )

        logger.info(
            "[TTS Engine] Rendered %d bytes via '%s' for %s",
            len(audio_bytes),
            rendered_by or "(none)",
            msg.message_uuid7,
        )

        # Return the audio (or empty audio when everything failed) on the same
        # message so it's persisted to the timeline and the stage acks. This
        # also acks the stage even in the failure case.
        reply = pb.MessagePostProcess(
            message_uuid7=msg.message_uuid7,
            processed_message=text_to_speak,
        )
        if audio_bytes:
            reply.audio = audio_bytes
            reply.audio_type = "audio/mpeg"
        await client.send("message_post_process", reply)
        mark_synthesized(msg.message_uuid7)

        # Optional local playback for standalone/no-display setups.
        if config.get("play_locally", False):
            try:
                volume = float(config.get("volume", 0.4))
            except (TypeError, ValueError):
                logger.warning("Invalid 'volume' in config; using 0.4.")
                volume = 0.4
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(thread_executor, play_audio, str(output_path), volume)

    # Connect + listen with automatic reconnect: a closed WebSocket ends
    # listen() and we reconnect with exponential backoff. The client performs a
    # fresh ConnectionRequest + auth on every connect, so re-instantiating it
    # per iteration re-authenticates cleanly.
    reconnect_backoff = float(config.get("reconnect_backoff_base", 1.0))
    max_backoff = float(config.get("reconnect_backoff_max", 30.0))
    backoff_reset_threshold = float(config.get("backoff_reset_threshold", 30))
    open_timeout = float(config.get("open_timeout", 10))
    handshake_timeout = float(config.get("handshake_timeout", 10))
    max_pending_tasks = int(config.get("max_pending_tasks", 16))
    while True:
        session_start = time.monotonic()
        client = None
        try:
            client = await (
                CockatielClient.connect(module_name)
                .endpoint(engine_ip, engine_port)
                .pin(pairing_pin)
                .position("postprocess")
                .open_timeout(open_timeout)
                .handshake_timeout(handshake_timeout)
                .max_pending_tasks(max_pending_tasks)
                .connect()
            )
            logger.info("TTS Service connected as '%s' (postprocess).", module_name)

            @client.on("message_post_process")
            async def handle_post_process(msg, container, _client=client):
                await handle_synthesis(_client, msg, container)

            logger.info("Listening for incoming Cockatiel engine stream payloads...")
            await client.listen()
            logger.warning("TTS Service connection to engine closed.")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("TTS Service connection error: %s", e)
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass

        # Reset the backoff after a healthy session; otherwise keep doubling.
        if time.monotonic() - session_start >= backoff_reset_threshold:
            reconnect_backoff = float(config.get("reconnect_backoff_base", 1.0))
        else:
            reconnect_backoff = min(reconnect_backoff * 2, max_backoff)
        logger.info("Reconnecting to engine in %.0fs...", reconnect_backoff)
        await asyncio.sleep(reconnect_backoff)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("TTS Service stopped by user.")
