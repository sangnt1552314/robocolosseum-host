"""GPU worker: connect outward to the user's server, infer, reply.

The model is loaded exactly once. The worker dials the user's callback URL over
WSS, authenticates with the per-session callback token, announces itself with
``worker_ready`` (carrying its ``session_id`` and model), then answers each
``observation`` with an ``action`` chunk. It reconnects with a bounded backoff
if the connection drops.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pbs_remote_inference.models import MODELS
from pbs_remote_inference.protocol import (
    Action,
    Observation,
    WorkerAck,
    WorkerError,
    WorkerReady,
    dump_message,
    parse_message,
)
from pbs_remote_inference.worker.observation import to_adapter_observation
from robocolosseum.config import CameraMapping, load_app_config
from robocolosseum.policies.registry import create_policy

log = logging.getLogger(__name__)


class Policy(Protocol):
    def predict(self, observation: Any) -> np.ndarray: ...


class DummyPolicy:
    """Model-free stand-in for networking tests (no torch, no checkpoint)."""

    name = "dummy"

    def __init__(self, action_horizon: int = 16, action_dim: int = 8) -> None:
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def load(self) -> None:
        log.info("Dummy policy ready (horizon=%d dim=%d)", self.action_horizon, self.action_dim)

    def predict(self, observation: Any) -> np.ndarray:
        base = np.linspace(0.0, 0.01, self.action_dim, dtype=np.float32)
        steps = np.arange(self.action_horizon, dtype=np.float32)[:, None]
        return base[None, :] * (steps + 1.0)


@dataclass
class LoadedPolicy:
    policy: Policy
    camera_mapping: CameraMapping
    model_name: str  # public name used on the wire
    policy_key: str  # registry key ("dummy" for the stand-in)
    action_horizon: int


def load_policy(model: str, config_path: str | None) -> LoadedPolicy:
    """Load the dummy policy or a real adapter (reusing the existing registry)."""
    if model == "dummy":
        policy = DummyPolicy()
        policy.load()
        return LoadedPolicy(policy, CameraMapping(), "dummy", "dummy", policy.action_horizon)

    if model not in MODELS:
        raise KeyError(f"Unknown model '{model}'. Known: {sorted(MODELS)}")
    spec = MODELS[model]
    path = Path(config_path) if config_path else Path(spec.config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Config '{path}' not found. Pass --config pointing at the policy YAML "
            f"for '{model}'."
        )
    app_config = load_app_config(path)
    log.info("Loading model '%s' (policy=%s) from %s", model, app_config.policy.name, path)
    adapter = create_policy(app_config.policy.name, app_config)
    adapter.load()
    log.info("Model ready: %s", model)
    return LoadedPolicy(
        adapter,
        app_config.camera_mapping,
        model,
        app_config.policy.name,
        app_config.runtime.max_horizon,
    )


@dataclass
class WorkerConfig:
    server_url: str
    token: str
    session_id: str
    worker_id: str = ""
    reconnect_delay: float = 3.0
    max_reconnect_delay: float = 30.0
    max_reconnects: int | None = None

    def __post_init__(self) -> None:
        if not self.worker_id:
            self.worker_id = f"worker-{self.session_id[-8:] or 'unknown'}"


def _predict(loaded: LoadedPolicy, observation: Observation) -> np.ndarray:
    if loaded.policy_key == "dummy":
        actions = loaded.policy.predict(observation)
    else:
        adapter_obs = to_adapter_observation(observation, loaded.camera_mapping)
        actions = loaded.policy.predict(adapter_obs)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(f"policy returned shape {actions.shape}, expected 2D")
    if not np.isfinite(actions).all():
        raise ValueError("policy produced NaN/Inf actions")
    return actions


async def _handle_observation(websocket, loaded: LoadedPolicy, observation: Observation) -> None:
    log.info("Observation received: request_id=%s", observation.request_id)
    start = time.perf_counter()
    try:
        actions = await asyncio.get_running_loop().run_in_executor(
            None, _predict, loaded, observation
        )
    except Exception as exc:  # inference failure -> report, never crash the loop
        log.exception("Inference failed for request_id=%s", observation.request_id)
        await websocket.send(
            dump_message(WorkerError(request_id=observation.request_id, message=str(exc)))
        )
        return
    inference_ms = (time.perf_counter() - start) * 1000.0
    log.info(
        "Inference done: request_id=%s horizon=%d dim=%d latency=%.1fms",
        observation.request_id,
        actions.shape[0],
        actions.shape[1],
        inference_ms,
    )
    await websocket.send(
        dump_message(
            Action(
                request_id=observation.request_id,
                model=loaded.model_name,
                actions=actions.tolist(),
                inference_time_ms=inference_ms,
            )
        )
    )


async def _run_once(config: WorkerConfig, loaded: LoadedPolicy) -> None:
    import websockets

    headers = {"Authorization": f"Bearer {config.token}"}
    log.info("Callback connecting: %s", config.server_url)
    async with websockets.connect(
        config.server_url,
        additional_headers=headers,
        max_size=16 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=30,
    ) as websocket:
        log.info("Callback connected")
        await websocket.send(
            dump_message(
                WorkerReady(
                    session_id=config.session_id,
                    model=loaded.model_name,
                    worker_id=config.worker_id,
                    action_horizon=loaded.action_horizon,
                )
            )
        )
        ack = parse_message(await websocket.recv())
        if not isinstance(ack, WorkerAck) or not ack.accepted:
            reason = getattr(ack, "message", "") or "not accepted"
            raise RuntimeError(f"server rejected worker_ready: {reason}")
        log.info("Handshake complete for session %s", config.session_id)

        async for raw in websocket:
            message = parse_message(raw)
            if isinstance(message, Observation):
                await _handle_observation(websocket, loaded, message)
            else:
                log.warning("Ignoring unexpected message: %s", type(message).__name__)


async def run_worker(config: WorkerConfig, loaded: LoadedPolicy) -> None:
    from websockets.exceptions import ConnectionClosed, InvalidStatus

    attempt = 0
    while True:
        try:
            await _run_once(config, loaded)
            attempt = 0
        except InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                log.error("Authentication rejected by server (401); check the token")
                return
            log.warning("Connection rejected (status=%s)", status)
        except (ConnectionClosed, OSError, asyncio.TimeoutError, RuntimeError) as exc:
            log.warning("Connection lost: %s", exc)

        attempt += 1
        if config.max_reconnects is not None and attempt > config.max_reconnects:
            log.error("Giving up after %d reconnect attempts", attempt - 1)
            return
        delay = min(config.reconnect_delay * attempt, config.max_reconnect_delay)
        log.info("Reconnecting in %.1fs (attempt %d)", delay, attempt)
        await asyncio.sleep(delay)
