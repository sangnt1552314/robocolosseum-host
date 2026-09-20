#!/usr/bin/env python3
"""Run the PBS GPU inference worker.

Launched by the model's PBS script with values passed via ``qsub -v``::

    python pbs_remote_inference/worker/run_worker.py \\
        --model "$MODEL" --server "$CALLBACK_URL" \\
        --token "$CALLBACK_TOKEN" --session-id "$SESSION_ID"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pbs_remote_inference.worker.inference_worker import (  # noqa: E402
    WorkerConfig,
    load_policy,
    run_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", ""),
        help="Public model name (e.g. molmoact2-droid) or 'dummy'.",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("CALLBACK_URL", ""),
        help="wss:// callback URL of the user's local server.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CALLBACK_TOKEN", ""),
        help="Per-session callback token.",
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("SESSION_ID", ""),
        help="Session id issued by the launcher.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override the policy YAML (default: the model's registered config).",
    )
    parser.add_argument("--reconnect-delay", type=float, default=3.0)
    parser.add_argument("--max-reconnects", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    if not args.model:
        raise SystemExit("A --model (or MODEL) is required.")
    if not args.server:
        raise SystemExit("A --server (or CALLBACK_URL) is required.")
    if not args.token:
        raise SystemExit("A --token (or CALLBACK_TOKEN) is required.")
    if not args.session_id:
        raise SystemExit("A --session-id (or SESSION_ID) is required.")

    loaded = load_policy(args.model, args.config)
    config = WorkerConfig(
        server_url=args.server,
        token=args.token,
        session_id=args.session_id,
        reconnect_delay=args.reconnect_delay,
        max_reconnects=args.max_reconnects,
    )
    try:
        asyncio.run(run_worker(config, loaded))
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Worker stopped by user")


if __name__ == "__main__":
    main()
