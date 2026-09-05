"""Policy-independent worker lifecycle.

The runner owns everything that is *not* model-specific: connecting to the
Colosseum router, the observation loop, action validation, the dry-run/send
switch, and the timeout logic that lets the PBS job release its GPU as soon as
the evaluation finishes.

Colosseum session semantics (verified against the SDK source):
* A new evaluation session appears as a change in ``observation.session_id`` ->
  we call ``adapter.reset()``.
* ``get_obs`` raises ``TimeoutError`` on timeout and ``SDKConnectionError`` on a
  router disconnect / error. RESET and SESSION_CLOSE are not delivered through
  ``get_obs``; after SESSION_CLOSE the SDK clears ``session_id`` but a blocked
  ``get_obs`` is not woken. We therefore always pass a timeout and treat
  "timeout with an empty ``session_id``" as a finished session.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .config import AppConfig, RouterConfig
from .policies.registry import create_policy
from .utils.timing import Timer
from .utils.validation import validate_action_chunk

log = logging.getLogger(__name__)

RECONNECT_DELAY_SECONDS = 5.0


class _EarlyDisconnect(Exception):
    """Router dropped before any evaluation session began."""


def run_worker(
    config: AppConfig, router: RouterConfig, *, enable_action: bool = False
) -> None:
    from colosseum_policy_server import ColosseumPolicySDK, SDKConnectionError

    log.info("Loading policy: %s", config.policy.name)
    adapter = create_policy(config.policy.name, config)
    adapter.load()

    try:
        _connect_and_serve(
            config, router, adapter, enable_action, ColosseumPolicySDK, SDKConnectionError
        )
    finally:
        adapter.close()
        log.info("Policy worker shutting down")


def _connect_and_serve(
    config, router, adapter, enable_action, ColosseumPolicySDK, SDKConnectionError
) -> None:
    deadline = time.monotonic() + config.runtime.startup_timeout_seconds

    while True:
        try:
            sdk = ColosseumPolicySDK(
                router_url=router.url,
                token=router.token,
                action_spaces=tuple(config.runtime.action_spaces),
                control_hz=config.runtime.control_hz,
                max_horizon=config.runtime.max_horizon,
            )
            log.info("Connecting to router")
            sdk.start()
        except SDKConnectionError as exc:
            _retry_or_raise(deadline, exc)
            continue

        log.info("Router connected as server_id=%s", sdk.server_id)
        try:
            _session_loop(sdk, adapter, config, enable_action, SDKConnectionError)
            return
        except _EarlyDisconnect as exc:
            _retry_or_raise(deadline, exc)
        finally:
            sdk.close()


def _retry_or_raise(deadline: float, exc: BaseException) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        log.error("Could not establish a session within the startup timeout: %s", exc)
        raise TimeoutError("startup timeout: no session established") from exc
    delay = min(RECONNECT_DELAY_SECONDS, remaining)
    log.warning("Router connection issue (%s); retrying in %.1fs", exc, delay)
    time.sleep(delay)


def _session_loop(sdk, adapter, config, enable_action, SDKConnectionError) -> None:
    startup_timeout = config.runtime.startup_timeout_seconds
    idle_timeout = config.runtime.idle_timeout_seconds
    have_session = False
    current_session = None

    log.info("Waiting for session")
    while True:
        timeout = idle_timeout if have_session else startup_timeout
        try:
            obs = sdk.get_obs(timeout=timeout)
        except TimeoutError:
            if not have_session:
                log.info("No session within startup timeout; exiting")
            elif sdk.session_id == "":
                log.info("Session finished")
            else:
                log.info("Idle timeout reached; exiting")
            return
        except SDKConnectionError as exc:
            if not have_session:
                raise _EarlyDisconnect(str(exc)) from exc
            log.info("Router disconnected / session ended: %s", exc)
            return

        if obs.session_id != current_session:
            current_session = obs.session_id
            have_session = True
            log.info("Session started: %s", obs.session_id)
            adapter.reset()

        expected_dim = None
        if sdk.robot is not None and sdk.selected_action_space:
            expected_dim = sdk.robot.action_spaces.get(sdk.selected_action_space)

        log.info(
            "Observation received: seq=%s control_step=%s", obs.sequence, obs.control_step
        )
        with Timer() as timer:
            action = adapter.predict(obs)
        action = validate_action_chunk(
            action, max_horizon=sdk.max_horizon, expected_dim=expected_dim
        )
        log.info("Inference: %.0f ms", timer.elapsed_ms)
        log.info(
            "Action chunk: shape=%s first=%s",
            action.shape,
            np.array2string(action[0], precision=4, max_line_width=120),
        )

        if enable_action:
            sdk.send_action(action, observation=obs)
            log.info("Action sent to router")
        else:
            log.info("Robot action disabled - dry run")
