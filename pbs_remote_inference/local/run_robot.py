#!/usr/bin/env python3
"""Collaborator entrypoint: request a model and run the local robot loop.

The user never needs Hopper/PBS/qsub access. This program:

1. starts the local WebSocket robot server,
2. generates a per-session callback token,
3. asks the launcher API to submit the GPU worker for the chosen model,
4. waits for the worker to connect back over the user's ngrok URL,
5. streams observations and executes validated action chunks locally.

Example::

    python pbs_remote_inference/local/run_robot.py \\
        --model molmoact2-droid \\
        --launcher-url https://MY-LAUNCHER.ngrok-free.app \\
        --callback-url wss://THEIR-ROBOT.ngrok-free.app/worker \\
        --launcher-token "$LAUNCHER_API_TOKEN" \\
        --port 8000 --max-steps 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import websockets  # noqa: E402

from pbs_remote_inference.local.launcher_client import (  # noqa: E402
    LauncherClient,
    LauncherError,
)
from pbs_remote_inference.local.robot_interface import DummyRobot  # noqa: E402
from pbs_remote_inference.local.robot_server import (  # noqa: E402
    RobotServer,
    RobotServerConfig,
)
from pbs_remote_inference.local.safety import SafetyLimits  # noqa: E402

log = logging.getLogger("run_robot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Public model name, e.g. molmoact2-droid.")
    parser.add_argument(
        "--launcher-url", default="", help="Base URL of the launcher API (https://...)."
    )
    parser.add_argument(
        "--callback-url",
        default="",
        help="Public wss:// URL of THIS machine's worker endpoint (via ngrok).",
    )
    parser.add_argument(
        "--offline-session-id",
        default="",
        help="Debug mode: skip the launcher and wait for a worker using this "
        "session id (start the worker by hand). Requires --callback-token.",
    )
    parser.add_argument(
        "--launcher-token",
        default=os.environ.get("LAUNCHER_API_TOKEN", ""),
        help="Launcher API key (or LAUNCHER_API_TOKEN).",
    )
    parser.add_argument(
        "--callback-token",
        default="",
        help="Per-session token (auto-generated if omitted).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--instruction", default="pick up the cup")
    parser.add_argument("--inference-timeout", type=float, default=5.0)
    parser.add_argument("--control-interval", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--expected-action-dim", type=int, default=None)
    parser.add_argument("--max-action-value", type=float, default=None)
    parser.add_argument("--min-action-value", type=float, default=None)
    parser.add_argument(
        "--keep-job",
        action="store_true",
        help="Do not cancel the PBS job when the local loop exits.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    offline = bool(args.offline_session_id)
    if offline:
        if not args.callback_token:
            raise SystemExit("--offline-session-id requires --callback-token.")
    else:
        if not args.launcher_url or not args.callback_url:
            raise SystemExit("--launcher-url and --callback-url are required.")
        if not args.launcher_token:
            raise SystemExit("A --launcher-token (or LAUNCHER_API_TOKEN) is required.")
    callback_token = args.callback_token or secrets.token_urlsafe(24)

    robot = DummyRobot(instruction=args.instruction)
    server = RobotServer(
        robot,
        RobotServerConfig(
            token=callback_token,
            expected_model=args.model,
            inference_timeout=args.inference_timeout,
            control_interval=args.control_interval,
            max_steps=args.max_steps,
            limits=SafetyLimits(
                expected_action_dim=args.expected_action_dim,
                min_value=args.min_action_value,
                max_value=args.max_action_value,
            ),
        ),
    )
    client = (
        None if offline else LauncherClient(args.launcher_url, args.launcher_token)
    )

    async with websockets.serve(
        server._handler,
        args.host,
        args.port,
        process_request=server._authenticate,
        max_size=16 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    ):
        log.info("Local robot server listening on ws://%s:%d", args.host, args.port)
        if offline:
            session_id = args.offline_session_id
            server.expected_session_id = session_id
            log.info(
                "Offline mode: waiting for a worker with session_id=%s "
                "(start run_worker.py by hand)",
                session_id,
            )
        else:
            log.info("Requesting model '%s' from launcher %s", args.model, args.launcher_url)
            try:
                job = await client.submit(args.model, args.callback_url, callback_token)
            except LauncherError as exc:
                raise SystemExit(f"Launcher submission failed: {exc}")
            session_id = job["session_id"]
            server.expected_session_id = session_id
            log.info(
                "Job requested: session_id=%s job_id=%s status=%s",
                session_id,
                job.get("job_id"),
                job.get("status"),
            )
            log.info("Waiting for the PBS worker to connect back...")

        try:
            await server.session_complete.wait()
        finally:
            log.info("Local loop finished after %d executed chunk(s)", server.executed)
            if client is not None and not args.keep_job:
                try:
                    await client.cancel(session_id)
                    log.info("Cancelled PBS job for session %s", session_id)
                except LauncherError as exc:
                    log.warning("Could not cancel job: %s", exc)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        log.info("Stopped by user")


if __name__ == "__main__":
    main()
