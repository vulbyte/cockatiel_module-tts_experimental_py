"""
lib_cockatiel.py
Python client library for connecting to the cockatiel engine.

Single-connection auth: the client connects once, sends a ConnectionRequest
with the PIN, and the engine replies with a ConnectionRequestReturn carrying a
JWT in `container.auth_token`. All subsequent messages use that token on the
same connection.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import ssl
import sys
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import websockets

logger = logging.getLogger("lib_cockatiel")

# ProcessPosition enum values (must match the .proto)
_PROCESS_POSITION = {
    "unspecified": 0,
    "preprocess": 1,
    "inprocess": 2,
    "postprocess": 3,
    "connection": 4,
}


def _load_pb_module():
    proto_path = Path(
        os.environ.get(
            "COCKATIEL_PROTO_PATH",
            str(Path(__file__).with_name("cockatiel_protobuf.proto")),
        )
    )
    if not proto_path.exists():
        raise FileNotFoundError(f"Could not find cockatiel_protobuf.proto at {proto_path}.")

    cache_dir = Path(os.environ.get("COCKATIEL_PB2_CACHE", str(proto_path.parent / ".cockatiel_pb2_cache")))
    cache_dir.mkdir(exist_ok=True)
    generated_path = cache_dir / "cockatiel_protobuf_pb2.py"

    if not generated_path.exists() or generated_path.stat().st_mtime < proto_path.stat().st_mtime:
        try:
            from grpc_tools import protoc
        except ImportError as e:
            raise ImportError("grpcio-tools required to compile .proto file.") from e

        logger.info("Compiling %s...", proto_path.name)
        result = protoc.main(["protoc", f"-I{proto_path.parent}", f"--python_out={cache_dir}", str(proto_path)])
        if result != 0:
            raise RuntimeError(f"protoc failed with exit code {result}")

    if str(cache_dir) not in sys.path:
        sys.path.insert(0, str(cache_dir))

    return importlib.import_module("cockatiel_protobuf_pb2")


pb = _load_pb_module()

_PAYLOAD_FIELD_BY_MESSAGE_NAME: Dict[str, str] = {
    field.message_type.name: field.name
    for field in pb.Container.DESCRIPTOR.oneofs_by_name["payload"].fields
}


class CockatielClientBuilder:
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._ip = "127.0.0.1"
        self._port = 9734
        self._pin = 0
        self._priority = 10
        self._process_position = "postprocess"

    def endpoint(self, ip: str, port: int) -> "CockatielClientBuilder":
        self._ip = ip
        self._port = port
        return self

    def pin(self, pin: int) -> "CockatielClientBuilder":
        self._pin = pin
        return self

    def priority(self, priority: int) -> "CockatielClientBuilder":
        self._priority = priority
        return self

    def position(self, position: str) -> "CockatielClientBuilder":
        self._process_position = position
        return self

    async def connect(self) -> "CockatielClient":
        # WSS when the supervisor points us at the engine's self-signed cert
        # (COCKATIEL_TLS_CERT) — the engine rejects plain ws://. The cert is
        # loaded as the trust root so the self-signed cert is accepted (pinned).
        tls_cert = os.environ.get("COCKATIEL_TLS_CERT", "").strip()
        scheme = "wss" if tls_cert else "ws"
        engine_ws_url = f"{scheme}://{self._ip}:{self._port}"
        logger.info("Connecting to Cockatiel Engine at %s...", engine_ws_url)

        ws_kwargs = {}
        if tls_cert:
            ctx = ssl.create_default_context(cafile=tls_cert)
            ws_kwargs["ssl"] = ctx
        # Bound the connect so the service's reconnect loop can back off instead
        # of hanging forever on an unreachable engine.
        ws_kwargs["open_timeout"] = 10.0
        ws = await websockets.connect(engine_ws_url, **ws_kwargs)
        logger.info("Connected to WebSocket! Sending authentication handshake...")

        pos_enum = _PROCESS_POSITION.get(self._process_position.lower(), 4)

        handshake_container = pb.Container(
            version=1,
            auth_token="",
            module_name=self._module_name,
            module_instance_uuid7="",
            connection_request=pb.ConnectionRequest(
                pin=self._pin,
                process_position=pos_enum,
                priority=self._priority,
                module_instance_uuid7="",
            ),
        )

        await ws.send(handshake_container.SerializeToString())
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        except asyncio.TimeoutError:
            await ws.close()
            raise ConnectionError("Authentication timed out: engine did not reply to handshake.")

        if isinstance(raw, str):
            await ws.close()
            raise ConnectionError("Authentication failed: Engine sent text instead of binary Protobuf.")

        resp = pb.Container()
        resp.ParseFromString(raw)

        if resp.WhichOneof("payload") != "connection_request_return":
            await ws.close()
            raise ConnectionRefusedError("Authentication rejected by engine.")

        auth_token = resp.auth_token
        if not auth_token:
            await ws.close()
            raise ConnectionRefusedError("Authentication rejected by engine: empty auth token.")

        assigned_uuid = resp.connection_request_return.module_instance_uuid7
        logger.info("Successfully authenticated with Cockatiel Engine!")
        return CockatielClient(ws, self._module_name, auth_token, assigned_uuid)


class CockatielClient:
    def __init__(self, ws, module_name: str, auth_token: str, instance_uuid7: str, max_pending_tasks: int = 16):
        self._ws = ws
        self._module_name = module_name
        self._auth_token = auth_token
        self._instance_uuid7 = instance_uuid7
        self._send_lock = asyncio.Lock()
        self._callbacks: Dict[str, Callable[[Any, Any], Union[Awaitable[None], None]]] = {}
        # Bounds the number of in-flight background handler tasks (e.g. slow TTS
        # syntheses) so a chat flood can't accumulate tasks without bound. When
        # at capacity the newest task is dropped with a log instead of queuing.
        self._max_pending_tasks = max_pending_tasks
        self._pending_tasks: "deque[asyncio.Task]" = deque()

    @property
    def auth_token(self) -> str:
        return self._auth_token

    @property
    def instance_uuid7(self) -> str:
        return self._instance_uuid7

    @staticmethod
    def connect(module_name: str) -> CockatielClientBuilder:
        return CockatielClientBuilder(module_name)

    async def send(self, request_type: str, payload_message: Any) -> None:
        message_name = type(payload_message).__name__
        field_name = _PAYLOAD_FIELD_BY_MESSAGE_NAME.get(message_name)
        if field_name is None:
            raise ValueError(f"{message_name} isn't a valid Container payload type.")

        container = pb.Container(
            version=1,
            auth_token=self._auth_token,
            module_name=self._module_name,
            module_instance_uuid7=self._instance_uuid7,
        )
        getattr(container, field_name).CopyFrom(payload_message)

        async with self._send_lock:
            await self._ws.send(container.SerializeToString())

    async def receive(self, listener_fn: Callable[[Any], Union[Awaitable[None], None]]) -> None:
        async for raw in self._ws:
            if isinstance(raw, str):
                continue
            container = pb.Container()
            try:
                container.ParseFromString(raw)
            except Exception:
                logger.exception("Failed to decode incoming Protobuf frame")
                continue

            result = listener_fn(container)
            if asyncio.iscoroutine(result):
                await result

    def on(self, payload_key: str, callback: Optional[Callable[[Any, Any], Union[Awaitable[None], None]]] = None) -> Any:
        """Register a handler for a payload type. Usable both as
        `client.on("key", fn)` and as a decorator `@client.on("key")`."""
        if callback is None:

            def register(fn: Callable[[Any, Any], Union[Awaitable[None], None]]) -> Callable[[Any, Any], Union[Awaitable[None], None]]:
                self._callbacks[payload_key] = fn
                return fn

            return register
        self._callbacks[payload_key] = callback

    async def listen(self) -> None:
        async def _dispatch(container: Any) -> None:
            payload_key = container.WhichOneof("payload")
            # Answer the engine's liveness probe: an AuthVerify request is
            # answered with our own auth token so a quiet module is never
            # severed as "unresponsive".
            if payload_key == "auth_verify":
                reply = pb.Container(
                    version=1,
                    auth_token=self._auth_token,
                    module_name=self._module_name,
                    module_instance_uuid7=self._instance_uuid7,
                )
                reply.auth_verify.cur_auth = self._auth_token
                await self._ws.send(reply.SerializeToString())
                return
            if payload_key and payload_key in self._callbacks:
                result = self._callbacks[payload_key](getattr(container, payload_key), container)
                if asyncio.iscoroutine(result):
                    if len(self._pending_tasks) >= self._max_pending_tasks:
                        logger.warning(
                            "Dropping new background task for '%s': %d already "
                            "pending (cap %d).",
                            payload_key,
                            len(self._pending_tasks),
                            self._max_pending_tasks,
                        )
                        result.close()
                        return
                    # Run slow handlers (e.g. a multi-second TTS synthesis) as
                    # background tasks so the receive loop keeps processing
                    # frames — a slow handler must not block liveness probes.
                    task = asyncio.create_task(result)
                    self._pending_tasks.append(task)
                    task.add_done_callback(self._on_pending_done)

        await self.receive(_dispatch)

    def _on_pending_done(self, task: asyncio.Task) -> None:
        if task in self._pending_tasks:
            self._pending_tasks.remove(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Background handler task failed: %s", exc)

    async def close(self) -> None:
        await self._ws.close()