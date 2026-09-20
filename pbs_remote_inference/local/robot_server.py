"""Local WebSocket robot server (the callback endpoint the GPU worker dials).

The user exposes this server publicly with ngrok; the PBS worker connects
outward to it. Authentication is the per-session callback token (bearer header);
the worker must also present the matching ``session_id`` (and model) in its
``worker_ready`` message. All safety, timeout and stale handling lives here.
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
from dataclasses import dataclass

from websockets.exceptions import ConnectionClosed

from pbs_remote_inference.local.robot_interface import ObservationData, RobotInterface
from pbs_remote_inference.local.safety import (
    SafetyLimits,
    UnsafeActionError,
    validate_action,
)
from pbs_remote_inference.protocol import (
    Action,
    Observation,
    WorkerAck,
    WorkerError,
    WorkerReady,
    dump_message,
    encode_jpeg,
    parse_message,
)

log = logging.getLogger(__name__)


@dataclass
class RobotServerConfig:
    token: str
    expected_model: str
    server_name: str = "robot-local-server"
    inference_timeout: float = 5.0
    control_interval: float = 0.0
    max_steps: int | None = None
    limits: SafetyLimits = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.limits is None:
            self.limits = SafetyLimits()


class RobotServer:
    def __init__(self, robot: RobotInterface, config: RobotServerConfig) -> None:
        self.robot = robot
        self.config = config
        # Set by the launcher client once the session exists (before the worker
        # can realistically connect).
        self.expected_session_id: str | None = None
        self.session_complete = asyncio.Event()
        self.executed = 0
        self._request_counter = 0
        self._worker_active = False

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"obs-{self._request_counter:06d}"

    async def _authenticate(self, connection, request):
        auth = request.headers.get("Authorization", "")
        if not self.config.token or auth != f"Bearer {self.config.token}":
            log.warning("Rejected worker connection with missing/invalid token")
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        return None

    def _build_observation(self, data: ObservationData, request_id: str) -> Observation:
        return Observation(
            request_id=request_id,
            timestamp=time.time(),
            instruction=data.instruction,
            robot_state={k: [float(x) for x in v] for k, v in data.robot_state.items()},
            cameras={name: encode_jpeg(img) for name, img in data.cameras.items()},
        )

    async def _handler(self, websocket) -> None:
        if self._worker_active:
            log.warning("A worker is already connected; refusing the new connection")
            await websocket.close(code=1013, reason="server busy")
            return

        try:
            ready = parse_message(await websocket.recv())
        except (ConnectionClosed, ValueError) as exc:
            log.warning("Handshake failed before worker_ready: %s", exc)
            return
        if not isinstance(ready, WorkerReady):
            log.warning("First message was not worker_ready (%s)", type(ready).__name__)
            await websocket.close(code=1002, reason="expected worker_ready")
            return
        if self.expected_session_id and ready.session_id != self.expected_session_id:
            log.warning(
                "Rejected worker: session_id mismatch (got %s)", ready.session_id
            )
            await websocket.send(
                dump_message(
                    WorkerAck(
                        session_id=ready.session_id,
                        accepted=False,
                        message="session_id mismatch",
                    )
                )
            )
            await websocket.close(code=1008, reason="session mismatch")
            return
        if ready.model != self.config.expected_model:
            log.warning("Rejected worker: model mismatch (got %s)", ready.model)
            await websocket.send(
                dump_message(
                    WorkerAck(
                        session_id=ready.session_id,
                        accepted=False,
                        message="model mismatch",
                    )
                )
            )
            await websocket.close(code=1008, reason="model mismatch")
            return

        self._worker_active = True
        log.info(
            "Worker connected and ready: session_id=%s model=%s worker_id=%s",
            ready.session_id,
            ready.model,
            ready.worker_id or "<none>",
        )
        await websocket.send(
            dump_message(WorkerAck(session_id=ready.session_id, accepted=True))
        )
        try:
            await self._control_loop(websocket, ready)
        except ConnectionClosed as exc:
            log.warning("Worker disconnected: %s", exc)
        finally:
            self._worker_active = False
            self.robot.safe_stop()
            self.session_complete.set()
            log.info("Worker session ended: session_id=%s", ready.session_id)

    async def _control_loop(self, websocket, ready: WorkerReady) -> None:
        step = 0
        while self.config.max_steps is None or step < self.config.max_steps:
            data = self.robot.get_observation()
            request_id = self._next_request_id()
            observation = self._build_observation(data, request_id)
            sent = time.monotonic()
            await websocket.send(dump_message(observation))
            log.info("Observation sent: request_id=%s", request_id)

            try:
                raw = await asyncio.wait_for(
                    websocket.recv(), timeout=self.config.inference_timeout
                )
            except asyncio.TimeoutError:
                log.error(
                    "Inference TIMEOUT for request_id=%s after %.1fs",
                    request_id,
                    self.config.inference_timeout,
                )
                self.robot.safe_stop()
                continue

            now = time.monotonic()
            message = parse_message(raw)
            if isinstance(message, WorkerError):
                log.error(
                    "Worker error for request_id=%s: %s",
                    message.request_id,
                    message.message,
                )
                self.robot.safe_stop()
                continue
            if not isinstance(message, Action):
                log.error("Unexpected message type: %s", type(message).__name__)
                self.robot.safe_stop()
                continue

            try:
                actions = validate_action(
                    message,
                    expected_request_id=request_id,
                    sent_monotonic=sent,
                    now_monotonic=now,
                    timeout_seconds=self.config.inference_timeout,
                    limits=self.config.limits,
                )
            except UnsafeActionError as exc:
                log.error("Rejected action for request_id=%s: %s", request_id, exc)
                self.robot.safe_stop()
                continue

            log.info(
                "Action received: request_id=%s horizon=%d dim=%d "
                "round_trip=%.1fms worker_inference=%.1fms",
                request_id,
                actions.shape[0],
                actions.shape[1],
                (now - sent) * 1000.0,
                message.inference_time_ms,
            )
            self.robot.execute_actions(actions)
            self.executed += 1
            step += 1
            if self.config.control_interval > 0:
                await asyncio.sleep(self.config.control_interval)
